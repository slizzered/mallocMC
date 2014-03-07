/*
ScatterAlloc: Massively Parallel Dynamic Memory Allocation for the GPU.
http://www.icg.tugraz.at/project/mvp

Copyright (C) 2012 Institute for Computer Graphics and Vision,
Graz University of Technology

Author(s):  Markus Steinberger - steinberger ( at ) icg.tugraz.at
Michael Kenzel - kenzel ( at ) icg.tugraz.at
Carlchristian Eckert - c.eckert ( at ) hzdr.de

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <cuda.h>

//replace the cuda malloc and free calls
#define SCATTERALLOC_OVERWRITE_MALLOC 1

//set the template arguments using SCATTERALLOC_HEAPARGS
// pagesize ... byter per page
// accessblocks ... number of superblocks
// regionsize ... number of regions for meta data structur
// wastefactor ... how much memory can be wasted per alloc (multiplicative factor)
// use_coalescing ... combine memory requests of within each warp
// resetfreedpages ... allow pages to be reused with a different size
#define SCATTERALLOC_HEAPARGS 4096*1024, 8, 16, 2, true, true

// each pointer in the datastructure will point to this many
// elements of type allocElem_t
#define ELEMS_PER_SLOT 750

//include the scatter alloc heap
#include <src/include/scatteralloc/heap_impl.cuh>
#include <src/include/scatteralloc/utils.h>

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <conio.h>
#endif

#include <iostream>
#include <stdio.h>
#include <typeinfo>
#include <vector>

// get a CUDA error and print it nicely
#define CUDA_CHECK(cmd) {cudaError_t error = cmd; \
  if(error!=cudaSuccess){\
    printf("<%s>:%i ",__FILE__,__LINE__);\
    printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}

// start kernel, wait for finish and check errors
#define CUDA_CHECK_KERNEL_SYNC(...) __VA_ARGS__;CUDA_CHECK(cudaDeviceSynchronize())

// global variable for verbosity, might change due to user input '--verbose'
bool verbose = false;

// the type of the elements to allocate
typedef unsigned long long allocElem_t;

bool run_heap_verification(const int, const size_t, const unsigned, const unsigned);
void parse_cmdline(const int, char**, size_t*, unsigned*, unsigned*);
void print_help(char**);


// used to create an empty stream for non-verbose output 
struct nullstream : std::ostream {
  nullstream() : std::ostream(0) { }
};

// uses global verbosity to switch between std::cout and a NULL-output
std::ostream& dout() {
  static nullstream n;
  return verbose ? std::cout : n;
}


/**
 * will do a basic verification of scatterAlloc.
 *
 * @param argv if -q or --quiet is supplied as a
 *        command line argument, verbosity will be reduced
 * 
 * @return will return 0 if the verification was successful,
 *         otherwise returns 1
 */
int main(int argc, char** argv){
  bool correct     = false;
  size_t heapInMB  = size_t(1024U*1U); //1GB
  unsigned threads = 128;
  unsigned blocks  = 64;

  try
  {
    parse_cmdline(argc, argv, &heapInMB, &threads, &blocks);

    int cuda_device = argc > 1 ? atoi(argv[1]) : 0;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, cuda_device);
    dout() << "Using device: " << deviceProp.name << std::endl;

    if( deviceProp.major < 2 ) {
      std::cerr << "This GPU with Compute Capability " << deviceProp.major 
        << "." << deviceProp.minor <<  " does not meet minimum requirements." << std::endl;
      std::cerr << "A GPU with Compute Capability >= 2.0 is required." << std::endl;
      return -2;
    }

    correct = run_heap_verification(cuda_device, heapInMB, threads, blocks);

    cudaDeviceReset();
  }
  catch (const std::exception& e)
  {
    std::cout << e.what()  << std::endl;
#ifdef WIN32
    while (!_kbhit());
#endif
    return -1;
  }
  catch (...)
  {
    std::cout << "unknown exception!" << std::endl;
#ifdef WIN32
    while (!_kbhit());
#endif
    return -1;
  }

  if(correct){
    std::cout << "\033[0;32mverification successful ✔\033[0m" << std::endl;
    return 0;
  }else{
    std::cerr << "\033[0;31mverification failed\033[0m" << std::endl;
    return 1;
  }
}


/**
 * will parse command line arguments
 *
 * for more details, see print_help()
 *
 * @param argc argc from main()
 * @param argv argv from main()
 * @param heapInMP will be filled with the heapsize, if given as a parameter
 * @param threads will be filled with number of threads, if given as a parameter
 * @param blocks will be filled with number of blocks, if given as a parameter
 */
void parse_cmdline(
    const int argc,
    char**argv,
    size_t *heapInMB,
    unsigned *threads,
    unsigned *blocks
    ){

  std::vector<std::pair<std::string, std::string> > parameters;

  // Parse Commandline, tokens are shaped like ARG=PARAM or ARG
  // This requires to use '=', if you want to supply a value with a parameter
  for (int i = 1; i < argc; ++i) {
    char* pos = strtok(argv[i], "=");
    std::pair < std::string, std::string > p(std::string(pos), std::string(""));
    pos = strtok(NULL, "=");
    if (pos != NULL) {
      p.second = std::string(pos);
    }
    parameters.push_back(p);
  }

  // go through all parameters that were found
  for (unsigned i = 0; i < parameters.size(); ++i) {
    std::pair < std::string, std::string > p = parameters.at(i);

    if (p.first == "-v" || p.first == "--verbose") {
      verbose = true;
    }

    if (p.first == "--threads") {
      *threads = atoi(p.second.c_str());
    }

    if (p.first == "--blocks") {
      *blocks = atoi(p.second.c_str());
    }

    if(p.first == "--heapsize") {
      *heapInMB = size_t(atoi(p.second.c_str()));
    }

    if(p.first == "-h" || p.first == "--help"){
      print_help(argv);
      exit(0);
    }
  }
}


/**
 * prints a helpful message about program use
 *
 * @param argv the argv-parameter from main, used to find the program name
 */
void print_help(char** argv){
  std::stringstream s;

  s << "SYNOPSIS:"                                              << std::endl;
  s << argv[0] << " [OPTIONS]"                                  << std::endl;
  s << ""                                                       << std::endl;
  s << "OPTIONS:"                                               << std::endl;
  s << "  -h, --help"                                           << std::endl; 
  s << "    Print this help message and exit"                   << std::endl; 
  s << ""                                                       << std::endl; 
  s << "  -v, --verbose"                                        << std::endl;
  s << "    Print information about parameters and progress"    << std::endl;
  s << ""                                                       << std::endl; 
  s << "  --threads=N"                                          << std::endl; 
  s << "    Set the number of threads per block (default 128)"  << std::endl; 
  s << ""                                                       << std::endl; 
  s << "  --blocks=N"                                           << std::endl; 
  s << "    Set the number of blocks in the grid (default 64)"  << std::endl; 
  s << ""                                                       << std::endl; 
  s << "  --heapsize=N"                                         << std::endl; 
  s << "    Set the heapsize to N Megabyte (default 1024)"      << std::endl; 

  std::cout << s.str();
}


/**
 * checks validity of memory for each single cell
 *
 * checks on a per thread basis, if the values written during 
 * allocation are still the same. Also calculates the sum over
 * all allocated values for a more in-depth verification that
 * could be done on the host
 *
 * @param data the data to verify
 * @param counter should be initialized with 0 and will
 *        be used to count how many verifications were
 *        already done
 * @param globalSum will be filled with the sum over all
 *        allocated values in the structure
 * @param nSlots the size of the datastructure
 * @param correct should be initialized with 1.
 *        Will change to 0, if there was a value that didn't match
 */
__global__ void check_content(
    allocElem_t** data,
    unsigned long long *counter,
    unsigned long long* globalSum, 
    const size_t nSlots,
    int* correct
    ){

  unsigned long long sum=0;
  while(true){
    size_t pos = atomicAdd(counter,1);
    if(pos >= nSlots){break;}
    const size_t offset = pos*ELEMS_PER_SLOT;
    for(size_t i=0;i<ELEMS_PER_SLOT;++i){
      if (static_cast<allocElem_t>(data[pos][i]) != static_cast<allocElem_t>(offset+i)){
        //printf("\nError in Kernel: data[%llu][%llu] is %#010x (should be %#010x)\n",
        //    pos,i,static_cast<allocElem_t>(data[pos][i]),allocElem_t(offset+i));
        atomicAnd(correct,0);
      }
      sum += static_cast<unsigned long long>(data[pos][i]);
    }
  }
  atomicAdd(globalSum,sum);
}


/**
 * checks validity of memory for each single cell
 *
 * checks on a per thread basis, if the values written during 
 * allocation are still the same.
 *
 * @param data the data to verify
 * @param counter should be initialized with 0 and will
 *        be used to count how many verifications were
 *        already done
 * @param nSlots the size of the datastructure
 * @param correct should be initialized with 1.
 *        Will change to 0, if there was a value that didn't match
 */
__global__ void check_content_fast(
    allocElem_t** data,
    unsigned long long *counter,
    const size_t nSlots,
    int* correct
    ){

  int c = 1;
  while(true){
    size_t pos = atomicAdd(counter,1);
    if(pos >= nSlots){break;}
    const size_t offset = pos*ELEMS_PER_SLOT;
    for(size_t i=0;i<ELEMS_PER_SLOT;++i){
      if (static_cast<allocElem_t>(data[pos][i]) != static_cast<allocElem_t>(offset+i)){
        c=0;
      }
    }
  }
  atomicAnd(correct,c);
}


/**
 * allocate a lot of small arrays and fill them
 *
 * Each array has the size ELEMS_PER_SLOT and the type allocElem_t. 
 * Each element will be filled with a number that is related to its
 * position in the datastructure.
 *
 * @param data the datastructure to allocate
 * @param counter should be initialized with 0 and will
 *        hold, how many allocations were done
 * @param globalSum will hold the sum of all values over all
 *        allocated structures (for verification purposes)
 */
__global__ void allocAll(
    allocElem_t** data, 
    unsigned long long* counter, 
    unsigned long long* globalSum
    ){

  unsigned long long sum=0;
  while(true){
    allocElem_t* p = new allocElem_t[ELEMS_PER_SLOT];
    if(p == NULL) break;

    size_t pos = atomicAdd(counter,1);
    const size_t offset = pos*ELEMS_PER_SLOT;
    for(size_t i=0;i<ELEMS_PER_SLOT;++i){
      p[i] = static_cast<allocElem_t>(offset + i);
      sum += static_cast<unsigned long long>(p[i]);
    }
    data[pos] = p;
  }

  atomicAdd(globalSum,sum);
}


/**
 * free all the values again
 *
 * @param data the datastructure to free
 * @param counter should be an empty space on device memory, 
 *        counts how many elements were freed
 * @param max the maximum number of elements to free
 */
__global__ void deallocAll(
    allocElem_t** data,
    unsigned long long* counter,
    const size_t nSlots
    ){

  while(true){
    size_t pos = atomicAdd(counter,1);
    if(pos >= nSlots) break;
    delete data[pos];
  }
}


/**
 * damages one element in the data
 *
 * With help of this function, you can verify that
 * the checks actually work as expected and can find
 * an error, if one should exist
 *
 * @param data the datastructure to damage
 */
__global__ void damageElement(allocElem_t** data){
  data[1][0] = static_cast<allocElem_t>(5*ELEMS_PER_SLOT - 1);
}


/**
 * wrapper function to allocate memory on device
 *
 * allocates memory with scatterAlloc. Returns the number of 
 * created elements as well as the sum of these elements
 *
 * @param d_testData the datastructure which will hold 
 *        pointers to the created elements
 * @param h_nSlots will be filled with the number of elements
 *        that were allocated
 * @param h_sum will be filled with the sum of all elements created
 * @param blocks the size of the CUDA grid
 * @param threads the number of CUDA threads per block
 */
void allocate(
    allocElem_t** d_testData, 
    unsigned long long* h_nSlots, 
    unsigned long long* h_sum,
    const unsigned blocks,
    const unsigned threads
    ){

  dout() << "allocating on device...";

  unsigned long long zero = 0;
  unsigned long long *d_sum;
  unsigned long long *d_nSlots;

  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMalloc((void**) &d_sum,sizeof(unsigned long long)));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMalloc((void**) &d_nSlots, sizeof(unsigned long long)));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(d_sum,&zero,sizeof(unsigned long long),cudaMemcpyHostToDevice));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(d_nSlots,&zero,sizeof(unsigned long long),cudaMemcpyHostToDevice));

  CUDA_CHECK_KERNEL_SYNC(allocAll<<<blocks,threads>>>(d_testData,d_nSlots,d_sum));

  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(h_sum,d_sum,sizeof(unsigned long long),cudaMemcpyDeviceToHost));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(h_nSlots,d_nSlots,sizeof(unsigned long long),cudaMemcpyDeviceToHost));
  cudaFree(d_sum);
  cudaFree(d_nSlots);
  dout() << "done" << std::endl;
}


/**
 * Wrapper function to verify allocation on device
 *
 * Generates the same number that was written into each position of
 * the datastructure during allocation and compares the values.
 *
 * @param d_testData the datastructure which holds 
 *        pointers to the elements you want to verify
 * @param nSlots the size of d_testData
 * @param blocks the size of the CUDA grid
 * @param threads the number of CUDA threads per block
 * @return true if the verification was successful, false otherwise
 */
bool verify(
    allocElem_t **d_testData,
    const unsigned long long nSlots,
    const unsigned blocks,
    const unsigned threads
    ){

  dout() << "verifying on device... ";

  const unsigned long long zero = 0;
  int  h_correct = 1;
  int* d_correct;
  unsigned long long *d_sum;
  unsigned long long *d_counter;
  
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMalloc((void**) &d_sum, sizeof(unsigned long long)));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMalloc((void**) &d_counter, sizeof(unsigned long long)));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMalloc((void**) &d_correct, sizeof(int)));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(d_sum,&zero,sizeof(unsigned long long),cudaMemcpyHostToDevice));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(d_counter,&zero,sizeof(unsigned long long),cudaMemcpyHostToDevice));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(d_correct,&h_correct,sizeof(int),cudaMemcpyHostToDevice));

  // can be replaced by a call to check_content_fast, 
  // if the gaussian sum (see below) is not used and you 
  // want to be a bit faster
  CUDA_CHECK_KERNEL_SYNC(check_content<<<blocks,threads>>>(
        d_testData,
        d_counter,
        d_sum,
        static_cast<size_t>(nSlots),
        d_correct
        ));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(&h_correct,d_correct,sizeof(int),cudaMemcpyDeviceToHost));

  // This only works, if the type "allocElem_t"
  // can hold all the IDs (usually unsigned long long)
  /*
  dout() << "verifying on host...";
  unsigned long long h_sum, h_counter;
  unsigned long long gaussian_sum = (ELEMS_PER_SLOT*nSlots * (ELEMS_PER_SLOT*nSlots-1))/2;
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(&h_sum,d_sum,sizeof(unsigned long long),cudaMemcpyDeviceToHost));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(&h_counter,d_counter,sizeof(unsigned long long),cudaMemcpyDeviceToHost));
  if(gaussian_sum != h_sum){
    dout() << "\nGaussian Sum doesn't match: is " << h_sum;
    dout() << " (should be " << gaussian_sum << ")" << std::endl;
    h_correct=false;
  }
  if(nSlots != h_counter-(blocks*threads)){
    dout() << "\nallocated number of elements doesn't match: is " << h_counter;
    dout() << " (should be " << nSlots << ")" << std::endl;
    h_correct=false;
  }
  */

  if(h_correct){
    dout() << "done" << std::endl;
  }else{
    dout() << "failed" << std::endl;
  }

  cudaFree(d_correct);
  cudaFree(d_sum);
  cudaFree(d_counter);
  return static_cast<bool>(h_correct);
}


/**
 * Verify the heap allocation of ScatterAlloc
 *
 * Allocates as much memory as the heap allows. Make sure that allocated
 * memory actually holds the correct values without corrupting them. Will
 * fill the datastructure with values that are relative to the index and
 * later evalute, if the values inside stayed the same after allocating all
 * memory
 *
 * @param cuda_device the index of 
 *        the graphics card to use
 * @return true if the verification was successful,
 *         false otherwise
 */
bool run_heap_verification(
    const int cuda_device,
    const size_t heapMB,
    const unsigned blocks,
    const unsigned threads
    ){

  cudaSetDevice(cuda_device);
  cudaSetDeviceFlags(cudaDeviceMapHost);

  const size_t heapSize         = size_t(1024U*1024U) * heapMB;
  const size_t slotSize         = sizeof(allocElem_t)*ELEMS_PER_SLOT;
  const size_t nPointers        = ceil(static_cast<float>(heapSize) / slotSize);
  const size_t maxSlots         = heapSize/slotSize;
  const size_t maxSpace         = maxSlots*slotSize + nPointers*sizeof(allocElem_t*);
  bool correct                  = true;
  const unsigned long long zero = 0;


  dout() << "ScatterAlloc:          " << "page     sblock region waste coalesc reset" << std::endl;
  if(verbose){
    printf("                       %d  %d      %d     %d     %d       %d\n",SCATTERALLOC_HEAPARGS);
  }
  dout() << "Gridsize:              "     << blocks                   << std::endl;
  dout() << "Blocksize:             "     << threads                  << std::endl;
  dout() << "Allocated elements:    "     << ELEMS_PER_SLOT << " x "  << sizeof(allocElem_t);
  dout() << "    Byte ("  << slotSize     << " Byte)"                 << std::endl;
  dout() << "Heap:                  "     << heapSize                 << " Byte";
  dout() << " (" << heapSize/pow(1024,2)  << " MByte)"                << std::endl; 
  dout() << "max space w/ pointers: "     << maxSpace                 << " Byte";
  dout() << " (" << maxSpace/pow(1024,2)  << " MByte)"                << std::endl;
  dout() << "maximum of elements:   "     << maxSlots                 << std::endl;

  // initializing the heap
  initHeap(heapSize); 
  allocElem_t** d_testData;
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMalloc((void**) &d_testData, nPointers*sizeof(allocElem_t*)));

  // allocating with scatterAlloc
  unsigned long long nAllocSlots = 0;
  unsigned long long sumAllocElems = 0;
  allocate(d_testData,&nAllocSlots,&sumAllocElems,blocks,threads);

  const float allocFrac = static_cast<float>(nAllocSlots)*100/maxSlots;
  const size_t wasted = heapSize - static_cast<size_t>(nAllocSlots) * slotSize;
  dout() << "allocated elements:    "   << nAllocSlots;
  dout() << " (" << allocFrac << "%)"   << std::endl;
  dout() << "wasted heap space:     "   << wasted << " Byte";
  dout() << " (" << wasted/pow(1024,2)  << " MByte)" << std::endl;

  // verifying on device
  correct = correct && verify(d_testData,nAllocSlots,blocks,threads);

  // damaging one cell
  dout() << "damaging of element... ";
  CUDA_CHECK_KERNEL_SYNC(damageElement<<<1,1>>>(d_testData));
  dout() << "done" << std::endl;

  // verifying on device 
  // THIS SHOULD FAIL (damage was done before!). Therefore, we must inverse the logic
  correct = correct && !verify(d_testData,nAllocSlots,blocks,threads);

  // release all memory
  dout() << "deallocation...        ";
  unsigned long long* d_dealloc_counter;
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMalloc((void**) &d_dealloc_counter, sizeof(unsigned long long)));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(d_dealloc_counter,&zero,sizeof(unsigned long long),cudaMemcpyHostToDevice));
  CUDA_CHECK_KERNEL_SYNC(deallocAll<<<blocks,threads>>>(d_testData,d_dealloc_counter,static_cast<size_t>(nAllocSlots)));
  cudaFree(d_dealloc_counter);
  cudaFree(d_testData);

  dout() << "done "<< std::endl;
  return correct;
}
