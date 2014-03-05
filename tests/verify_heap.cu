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
#include <vector>

#define CUDA_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}
/*start kernel, wait for finish and check errors*/
#define CUDA_CHECK_KERNEL_SYNC(...) __VA_ARGS__;CUDA_CHECK(cudaDeviceSynchronize())

typedef GPUTools::uint32 uint;
typedef size_t allocatedElement;

bool run_heap_verification(int cuda_device);


int main(int argc, char** argv)
{
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
    std::cout << "\033[0;32mverification successful\033[0m" << std::endl;
    return 0;
  }else{
    std::cerr << "\033[0;31mverification failed\033[0m" << std::endl;
    return 1;
  }
}


__global__ void checkIfRunning(){
  printf("still running!\n");
}





/**
 * @brief checks on a per thread basis, if the gaussian sum is correct
 *
 */
__global__ void check_idfilling(allocatedElement** array,size_t *counter,allocatedElement* globalSum, size_t max){
  allocatedElement sum=0;
  while(true){
    allocatedElement pos = atomicAdd((unsigned long long*)counter,1);
    if(pos >= max){
      break;
    }
    if(*(array[pos]) != (allocatedElement) pos){
      printf("\nError in Kernel: array[%llu] is %llu (should be %llu)\n",pos,*(array[pos]),pos);
    }
    sum += *(array[pos]);
  }

  atomicAdd((unsigned long long*)globalSum,(unsigned long long)sum);

}

/**
 * @brief allocate a lot of small arrays
 *
 *
 */
__global__ void allocAll(allocatedElement** array, size_t* counter, allocatedElement* globalSum){
  allocatedElement sum=0;

  while(true){
    allocatedElement* p = new allocatedElement[ALLOCATION_SIZE];
    if(p == NULL) break;

    allocatedElement pos = atomicAdd((unsigned long long*)counter,1);
    array[pos] = p;
    *(array[pos]) = (allocatedElement) pos;
    sum += pos;
  }

  //TODO: check for maximum value
  atomicAdd((unsigned long long*)globalSum,(unsigned long long)sum);
}

/**
 * @brief free all the values again
 *
 */
__global__ void deAllocAll(allocatedElement** array,size_t* counter, size_t max){
  while(true){
    allocatedElement pos = atomicAdd((unsigned long long*)counter,1);
    if(pos >= max) break;
    delete array[pos];
  }
}

__global__ void damageElement(allocatedElement** array){
  *(array[5]) = 4;
}


void allocate(allocatedElement** d_testData,allocatedElement* h_numberOfElements, allocatedElement* h_sum,const unsigned blocks,const unsigned threads){
  allocatedElement zero = 0;
  allocatedElement *d_sum, *d_numberOfElements;

  std::cout << "allocating the whole heap...";
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMalloc((void**) &d_sum,sizeof(allocatedElement)));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMalloc((void**) &d_numberOfElements, sizeof(allocatedElement)));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(d_sum,&zero,sizeof(allocatedElement),cudaMemcpyHostToDevice));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(d_numberOfElements,&zero,sizeof(allocatedElement),cudaMemcpyHostToDevice));

  CUDA_CHECK_KERNEL_SYNC(allocAll<<<blocks,threads>>>(d_testData,d_numberOfElements,d_sum));

  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(h_sum,d_sum,sizeof(allocatedElement),cudaMemcpyDeviceToHost));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(h_numberOfElements,d_numberOfElements,sizeof(allocatedElement),cudaMemcpyDeviceToHost));
  cudaFree(d_sum);
  cudaFree(d_numberOfElements);
  std::cout << "\r";
}

bool verify(allocatedElement **d_testData,const size_t numberOfElements){

  const unsigned blocks = 64;
  const unsigned threads = 64;
  const allocatedElement zero = 0;
  bool correct = true;
  allocatedElement *d_sum, *d_counter;
  allocatedElement h_sum, h_counter;



  std::cout << "verifying on device...";
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMalloc((void**) &d_sum, sizeof(allocatedElement)));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMalloc((void**) &d_counter, sizeof(allocatedElement)));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(d_sum,&zero,sizeof(allocatedElement)*1,cudaMemcpyHostToDevice));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(d_counter,&zero,sizeof(allocatedElement)*1,cudaMemcpyHostToDevice));
  CUDA_CHECK_KERNEL_SYNC(check_idfilling<<<blocks,threads>>>(d_testData,d_counter,d_sum,numberOfElements));
  std::cout << "done" << std::endl;


  std::cout << "verifying on host...";
  allocatedElement gaussian_sum = (numberOfElements * (numberOfElements-1))/2;
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(&h_sum,d_sum,sizeof(allocatedElement),cudaMemcpyDeviceToHost));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(&h_counter,d_counter,sizeof(allocatedElement),cudaMemcpyDeviceToHost));
  if(gaussian_sum != h_sum){
    std::cerr << "\nGaussian Sum doesn't match: is " << h_sum << " (should be " << gaussian_sum << ")" << std::endl;
    correct=false;;
  }
  if(numberOfElements != h_counter-(blocks*threads)){
    std::cerr << "\nallocated number of elements doesn't match: is " << h_counter << " (should be " << numberOfElements << ")" << std::endl;
    correct=false;;
  }
  cudaFree(d_sum);
  cudaFree(d_counter);
  if(correct){
    std::cout << "successful                  " << std::endl;
  }else{
    std::cerr << "failed                      " << std::endl;
  }
  return correct;
}


/**
 * @brief verify that the heap actually holds the correct values without corrupting them
 *
 */
bool run_heap_verification(int cuda_device)
{
  cudaSetDevice(cuda_device);
  cudaSetDeviceFlags(cudaDeviceMapHost);

  const unsigned blocks         = 64; 
  const unsigned threads        = 128;

  const allocatedElement zero   = 0;
  const size_t heapSize         = size_t(2U)*size_t(1024U*1024U*1024U);
  const size_t elemSize         = sizeof(allocatedElement)*ALLOCATION_SIZE;
  const size_t nPointers        = ceil(float(heapSize) / elemSize); //maybe the same as maxElements? 
  const size_t maxElements      = heapSize/elemSize;
  bool correct                  = true;


  std::cout << "ScatterAlloc:          " << "page     sblock region waste coalesc reset" << std::endl;
  printf(      "                       %d  %d      %d     %d     %d       %d\n",SCATTERALLOC_HEAPARGS);

  std::cout << "Gridsize:              " << blocks << std::endl;
  std::cout << "Blocksize:             " << threads << std::endl;
  std::cout << "Allocated elements:    " << elemSize << " Byte" << std::endl;
  std::cout << "Heap:                  " << heapSize << " Byte";
  std::cout << " (" << heapSize/pow(1024,2) << " MByte)" << std::endl; 

  const allocatedElement maxSpace = maxElements*elemSize + nPointers*sizeof(allocatedElement*);
  std::cout << "max space w/ pointers: " << maxSpace << " Byte";
  std::cout << " (" << maxSpace/pow(1024,2) << " MByte)" << std::endl;

  std::cout << "maximum of elements:   " << maxElements << std::endl;

  // initializing the heap
  initHeap(heapSize); 
  allocatedElement** d_testData;
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMalloc((void**) &d_testData, (heapSize/sizeof(allocatedElement))*sizeof(allocatedElement*)));


  // allocating with scatterAlloc
  size_t numberOfAllocatedElements = 0;
  allocatedElement sumOfAllocatedElements = 0;
  allocate(d_testData,&numberOfAllocatedElements,&sumOfAllocatedElements,blocks,threads);
  std::cout << "allocated elements:    " << numberOfAllocatedElements;
  const float allocElementsPercentage = float(100*numberOfAllocatedElements)/maxElements;
  std::cout << " (" << allocElementsPercentage << "%)" << std::endl;

  const size_t wastedHeap = heapSize - numberOfAllocatedElements * elemSize ;
  std::cout << "wasted heap space:     " << wastedHeap << " Byte";
  std::cout << " (" << wastedHeap/pow(1024,2) << " MByte)" << std::endl;

  // verifying on device
  correct = correct && verify(d_testData,numberOfAllocatedElements);

  // damaging one cell
  std::cout << "damaging of element...";
  CUDA_CHECK_KERNEL_SYNC(damageElement<<<1,1>>>(d_testData));
  std::cout << "done" << std::endl;

  // verifying on device 
  // THIS SHOULD FAIL (damage was done before!). Therefore, we must inverse the logic
  correct = correct && !verify(d_testData,numberOfAllocatedElements);

  // release all memory
  std::cout << "deallocation...";
  size_t* d_dealloc_counter;
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMalloc((void**) &d_dealloc_counter, sizeof(size_t)));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(d_dealloc_counter,&zero,sizeof(size_t),cudaMemcpyHostToDevice));
  CUDA_CHECK_KERNEL_SYNC(deAllocAll<<<blocks,threads>>>(d_testData,d_dealloc_counter,numberOfAllocatedElements));
  cudaFree(d_dealloc_counter);
  cudaFree(d_testData);
  std::cout << "done "<< std::endl;

  return correct;

}
