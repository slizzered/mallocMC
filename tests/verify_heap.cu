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
#define ALLOCATION_SIZE 750

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

#define CUDA_CHECK(cmd) {cudaError_t error = cmd; \
  if(error!=cudaSuccess){\
    printf("<%s>:%i ",__FILE__,__LINE__);\
    printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}
/*start kernel, wait for finish and check errors*/
#define CUDA_CHECK_KERNEL_SYNC(...) __VA_ARGS__;CUDA_CHECK(cudaDeviceSynchronize())

typedef GPUTools::uint32 uint;
typedef int8_t allocatedElement;

bool run_heap_verification(int cuda_device);


int main(int argc, char** argv){
  bool correct = false;
  try
  {
    int cuda_device = argc > 1 ? atoi(argv[1]) : 0;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, cuda_device);
    std::cout << "Using device: " << deviceProp.name << std::endl;

    if( deviceProp.major < 2 ) {
      std::cerr << "This GPU with Compute Capability " << deviceProp.major 
        << "." << deviceProp.minor <<  " does not meet minimum requirements." << std::endl;
      std::cerr << "A GPU with Compute Capability >= 2.0 is required." << std::endl;
      return -2;
    }

    correct = run_heap_verification(cuda_device);

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
 * @brief checks on a per thread basis, if the gaussian sum is correct
 *
 */
__global__ void check_idfilling(
    allocatedElement** array,
    unsigned long long *counter,
    unsigned long long* globalSum, 
    const size_t max,
    int* correct
    ){

  unsigned long long sum=0;
  while(true){
    size_t pos = atomicAdd(counter,1);
    if(pos >= max){break;}
    const size_t offset = pos*ALLOCATION_SIZE;
    for(size_t i=0;i<ALLOCATION_SIZE;++i){
      if (static_cast<allocatedElement>(array[pos][i]) != static_cast<allocatedElement>(offset+i)){
        //printf("\nError in Kernel: array[%llu][%llu] is %#010x (should be %#010x)\n",
        //    pos,i,static_cast<allocatedElement>(array[pos][i]),allocatedElement(offset+i));
        atomicAnd(correct,0);
      }
      sum += static_cast<unsigned long long>(array[pos][i]);
    }
  }

  atomicAdd(globalSum,sum);

}

__global__ void check_idfilling_fast(
    allocatedElement** array,
    unsigned long long *counter,
    const size_t max,
    int* correct
    ){

  int c = 1;
  while(true){
    size_t pos = atomicAdd(counter,1);
    if(pos >= max){break;}
    const size_t offset = pos*ALLOCATION_SIZE;
    for(size_t i=0;i<ALLOCATION_SIZE;++i){
      if (static_cast<allocatedElement>(array[pos][i]) != static_cast<allocatedElement>(offset+i)){
        c=0;
      }
    }
  }
  atomicAnd(correct,c);
}

/**
 * @brief allocate a lot of small arrays
 *
 */
__global__ void allocAll(
    allocatedElement** array, 
    unsigned long long* counter, 
    unsigned long long* globalSum
    ){

  unsigned long long sum=0;
  while(true){
    allocatedElement* p = new allocatedElement[ALLOCATION_SIZE];
    if(p == NULL) break;

    size_t pos = atomicAdd(counter,1);
    const size_t offset = pos*ALLOCATION_SIZE;
    for(size_t i=0;i<ALLOCATION_SIZE;++i){
      p[i] = static_cast<allocatedElement>(offset + i);
      sum += static_cast<unsigned long long>(p[i]);
    }
    array[pos] = p;
  }

  atomicAdd(globalSum,sum);
}


/**
 * @brief free all the values again
 *
 * @param array the datastructure to free
 * @param counter should be an empty space on device memory, 
 *        counts how many elements were freed
 * @param max the maximum number of elements to free
 */
__global__ void deAllocAll(
    allocatedElement** array,
    unsigned long long* counter,
    const size_t max
    ){

  while(true){
    size_t pos = atomicAdd(counter,1);
    if(pos >= max) break;
    delete array[pos];
  }
}


/**
 * @brief damages one element in the array, so you 
 *        can see if your checks actually work
 *
 * @param array the datastructure to damage
 */
__global__ void damageElement(allocatedElement** array){
  array[1][0] = static_cast<allocatedElement>(5*ALLOCATION_SIZE - 1);
}


/**
 * @brief wrapper function to allocate some memory on the device
 *        with scatterAlloc. Returns the number of created elements as well
 *        as the sum of these elements
 *
 * @param d_testData the datastructure which will hold 
 *        pointers to the created elements
 * @param h_numberOfElements will be filled with the number of elements
 *        that were allocated
 * @param h_sum will be filled with the sum of all elements created
 * @param blocks the size of the CUDA grid
 * @param threads the number of CUDA threads per block
 */
void allocate(
    allocatedElement** d_testData, 
    unsigned long long* h_numberOfElements, 
    unsigned long long* h_sum,
    const unsigned blocks,
    const unsigned threads
    ){

  unsigned long long zero = 0;
  unsigned long long *d_sum;
  unsigned long long *d_numberOfElements;

  std::cout << "allocating on device...";
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMalloc((void**) &d_sum,sizeof(unsigned long long)));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMalloc((void**) &d_numberOfElements, sizeof(unsigned long long)));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(d_sum,&zero,sizeof(unsigned long long),cudaMemcpyHostToDevice));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(d_numberOfElements,&zero,sizeof(unsigned long long),cudaMemcpyHostToDevice));

  CUDA_CHECK_KERNEL_SYNC(allocAll<<<blocks,threads>>>(d_testData,d_numberOfElements,d_sum));

  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(h_sum,d_sum,sizeof(unsigned long long),cudaMemcpyDeviceToHost));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(h_numberOfElements,d_numberOfElements,sizeof(unsigned long long),cudaMemcpyDeviceToHost));
  cudaFree(d_sum);
  cudaFree(d_numberOfElements);
  std::cout << "done" << std::endl;
}


/**
 * @brief wrapper function to verify some allocated memory on the device
 *
 * @param d_testData the datastructure which holds 
 *        pointers to the elements you want to verify
 * @param h_numberOfElements the size of d_testData
 * @return true if the verification was successful, false otherwise
 */
bool verify(allocatedElement **d_testData,const unsigned long long numberOfElements){

  int h_correct = 1;
  const unsigned blocks = 64;
  const unsigned threads = 64;
  const unsigned long long   zero = 0;
  unsigned long long *d_sum, *d_counter;
  int* d_correct;
  
  std::cout << "verifying on device... ";
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMalloc((void**) &d_sum, sizeof(unsigned long long)));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMalloc((void**) &d_counter, sizeof(unsigned long long)));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMalloc((void**) &d_correct, sizeof(int)));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(d_sum,&zero,sizeof(unsigned long long),cudaMemcpyHostToDevice));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(d_counter,&zero,sizeof(unsigned long long),cudaMemcpyHostToDevice));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(d_correct,&h_correct,sizeof(int),cudaMemcpyHostToDevice));
  CUDA_CHECK_KERNEL_SYNC(check_idfilling_fast<<<blocks,threads>>>(
        d_testData,
        d_counter,
        //d_sum,
        static_cast<size_t>(numberOfElements),
        d_correct
        ));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(&h_correct,d_correct,sizeof(int),cudaMemcpyDeviceToHost));

  //std::cout << "verifying on host...";
  //unsigned long long h_sum, h_counter;
  //unsigned long long gaussian_sum = (ALLOCATION_SIZE*numberOfElements * (ALLOCATION_SIZE*numberOfElements-1))/2;
  //SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(&h_sum,d_sum,sizeof(unsigned long long),cudaMemcpyDeviceToHost));
  //SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(&h_counter,d_counter,sizeof(unsigned long long),cudaMemcpyDeviceToHost));
  //if(gaussian_sum != h_sum){
  //  std::cerr << "\nGaussian Sum doesn't match: is " << h_sum;
  //  std::cerr << " (should be " << gaussian_sum << ")" << std::endl;
  //  h_correct=false;
  //}
  //if(numberOfElements != h_counter-(blocks*threads)){
  //  std::cerr << "\nallocated number of elements doesn't match: is " << h_counter;
  //  std::cerr << " (should be " << numberOfElements << ")" << std::endl;
  //  h_correct=false;
  //}

  cudaFree(d_correct);
  cudaFree(d_sum);
  cudaFree(d_counter);
  if(h_correct){
    std::cout << "done                        " << std::endl;
  }else{
    std::cerr << "failed                      " << std::endl;
  }
  return static_cast<bool>(h_correct);
}


/**
 * @brief verify that the heap actually holds the correct values without corrupting them
 * @param cuda_device the index of the graphics card to use
 * @return true if the verification was successful, false otherwise
 */
bool run_heap_verification(int cuda_device){
  cudaSetDevice(cuda_device);
  cudaSetDeviceFlags(cudaDeviceMapHost);

  const unsigned blocks     = 64; 
  const unsigned threads    = 128;

  const size_t heapSize     = size_t(4U)*size_t(1024U*1024U*1024U); //4GB
  const size_t elemSize     = sizeof(allocatedElement)*ALLOCATION_SIZE;
  const size_t nPointers    = ceil(static_cast<float>(heapSize) / elemSize);
  const size_t maxElements  = heapSize/elemSize;
  const size_t maxSpace     = maxElements*elemSize + nPointers*sizeof(allocatedElement*);
  bool correct              = true;
  const unsigned long long zero         = 0;


  std::cout << "ScatterAlloc:       " << "page     sblock region waste coalesc reset" << std::endl;
  printf(      "                    %d  %d      %d     %d     %d       %d\n",SCATTERALLOC_HEAPARGS);
  std::cout << "Gridsize:              " << blocks << std::endl;
  std::cout << "Blocksize:             " << threads << std::endl;
  std::cout << "Allocated elements:    " << ALLOCATION_SIZE << " x " << sizeof(allocatedElement);
  std::cout << "   Byte (" << elemSize << " Byte)" << std::endl;
  std::cout << "Heap:                  " << heapSize << " Byte";
  std::cout << " (" << heapSize/pow(1024,2) << " MByte)" << std::endl; 
  std::cout << "max space w/ pointers: " << maxSpace << " Byte";
  std::cout << " (" << maxSpace/pow(1024,2) << " MByte)" << std::endl;
  std::cout << "maximum of elements:   " << maxElements << std::endl;

  // initializing the heap
  initHeap(heapSize); 
  allocatedElement** d_testData;
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMalloc((void**) &d_testData, nPointers*sizeof(allocatedElement*)));

  // allocating with scatterAlloc
  unsigned long long numberOfAllocatedElements = 0;
  unsigned long long sumOfAllocatedElements = 0;
  allocate(d_testData,&numberOfAllocatedElements,&sumOfAllocatedElements,blocks,threads);
  std::cout << "allocated elements:    " << numberOfAllocatedElements;
  const float allocElementsPercentage = float(100*numberOfAllocatedElements)/maxElements;
  std::cout << " (" << allocElementsPercentage << "%)" << std::endl;

  const size_t wastedHeap = heapSize - static_cast<size_t>(numberOfAllocatedElements) * elemSize;
  std::cout << "wasted heap space:     " << wastedHeap << " Byte";
  std::cout << " (" << wastedHeap/pow(1024,2) << " MByte)" << std::endl;

  // verifying on device
  correct = correct && verify(d_testData,numberOfAllocatedElements);

  // damaging one cell
  std::cout << "damaging of element... ";
  CUDA_CHECK_KERNEL_SYNC(damageElement<<<1,1>>>(d_testData));
  std::cout << "done" << std::endl;

  // verifying on device 
  // THIS SHOULD FAIL (damage was done before!). Therefore, we must inverse the logic
  correct = correct && !verify(d_testData,numberOfAllocatedElements);

  // release all memory
  std::cout << "deallocation...        ";
  unsigned long long* d_dealloc_counter;
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMalloc((void**) &d_dealloc_counter, sizeof(unsigned long long)));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(d_dealloc_counter,&zero,sizeof(unsigned long long),cudaMemcpyHostToDevice));
  CUDA_CHECK_KERNEL_SYNC(deAllocAll<<<blocks,threads>>>(d_testData,d_dealloc_counter,static_cast<size_t>(numberOfAllocatedElements)));
  cudaFree(d_dealloc_counter);
  cudaFree(d_testData);
  std::cout << "done "<< std::endl;

  return correct;
}
