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
typedef unsigned long long allocatedElement;

void run_heap_verification(int cuda_device);


int main(int argc, char** argv)
{
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

    run_heap_verification(cuda_device);

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

  return 0;
}


__global__ void checkIfRunning(){
  printf("still running!\n");
}





/**
 * @brief checks on a per thread basis, if the gaussian sum is correct
 *
 */
__global__ void check_idfilling(allocatedElement** array,unsigned long long *counter,unsigned long long* globalSum, unsigned long long max){
  allocatedElement sum=0;
  while(true){
    unsigned long long pos = atomicAdd(counter,1);
    if(pos >= max){
      break;
    }
    if(*(array[pos]) != (allocatedElement) pos){
      printf("\nError in Kernel: array[%llu] is %llu (should be %llu)\n",pos,*(array[pos]),pos);
    }
    sum += *(array[pos]);
  }

  atomicAdd(globalSum,sum);

}

/**
 * @brief allocate a lot of small arrays
 *
 *
 */
__global__ void allocAll(allocatedElement** array, unsigned long long* counter, unsigned long long* globalSum){
  allocatedElement sum=0;

  while(true){
    allocatedElement* p = new allocatedElement[ALLOCATION_SIZE];
    if(p == NULL) break;

    unsigned long long pos = atomicAdd(counter,1);
    array[pos] = p;
    *(array[pos]) = (allocatedElement) pos;
    sum += pos;
  }

  //TODO: check for maximum value
  atomicAdd(globalSum,sum);
}

/**
 * @brief free all the values again
 *
 */
__global__ void deAllocAll(allocatedElement** array,unsigned long long* counter, unsigned long long max){
 // while(true){
    //unsigned long long pos = atomicAdd(&counter[0],1);
    //if(pos >= max) break;
    //delete array[pos];
  //}
}

__global__ void damageElement(allocatedElement** array){
    *(array[5]) = 4;
}


void allocate(allocatedElement** d_testData,unsigned long long* h_numberOfElements, unsigned long long* h_sum,unsigned blocks, unsigned threads){
  unsigned long long zero = 0;
  unsigned long long *d_sum, *d_numberOfElements;

  std::cout << "allocating the whole heap...";
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMalloc((void**) &d_sum,sizeof(unsigned long long)));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMalloc((void**) &d_numberOfElements, sizeof(unsigned long long)));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(d_sum,&zero,sizeof(unsigned long long),cudaMemcpyHostToDevice));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(d_numberOfElements,&zero,sizeof(unsigned long long),cudaMemcpyHostToDevice));

  CUDA_CHECK_KERNEL_SYNC(allocAll<<<blocks,threads>>>(d_testData,d_numberOfElements,d_sum));

  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(h_sum,d_sum,sizeof(unsigned long long),cudaMemcpyDeviceToHost));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(h_numberOfElements,d_numberOfElements,sizeof(unsigned long long),cudaMemcpyDeviceToHost));
  cudaFree(d_sum);
  cudaFree(d_numberOfElements);
  std::cout << "\r";
}

void verify(allocatedElement **d_testData,unsigned long long numberOfElements){


  unsigned blocks = 64;
  unsigned threads = 64;
  unsigned long long zero = 0;
  unsigned long long *d_sum, *d_counter;
  unsigned long long h_sum, h_counter;

  std::cout << "verifying allocation on device...";
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMalloc((void**) &d_sum, sizeof(unsigned long long)));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMalloc((void**) &d_counter, sizeof(unsigned long long)));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(d_sum,&zero,sizeof(unsigned long long)*1,cudaMemcpyHostToDevice));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(d_counter,&zero,sizeof(unsigned long long)*1,cudaMemcpyHostToDevice));
  CUDA_CHECK_KERNEL_SYNC(check_idfilling<<<blocks,threads>>>(d_testData,d_counter,d_sum,numberOfElements));
  std::cout << "done" << std::endl;


  // verifying on host
  std::cout << "verifying allocation on host...";
  unsigned long long gaussian_sum = (numberOfElements * (numberOfElements-1))/2;
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(&h_sum,d_sum,sizeof(unsigned long long),cudaMemcpyDeviceToHost));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(&h_counter,d_counter,sizeof(unsigned long long),cudaMemcpyDeviceToHost));
  if(gaussian_sum != h_sum){
    std::cerr << "\nGaussian Sum doesn't match: is " << h_sum << " (should be " << gaussian_sum << ")" << std::endl;
  }
  if(numberOfElements != h_counter-(blocks*threads)){
    std::cerr << "\nallocated number of elements doesn't match: is " << h_counter << " (should be " << numberOfElements << ")" << std::endl;
  }
  cudaFree(d_sum);
  cudaFree(d_counter);
  std::cout << "done" << std::endl;
}

/**
 * @brief verify that the heap actually holds the correct values without corrupting them
 *
 */
void run_heap_verification(int cuda_device)
{
  cudaSetDevice(cuda_device);
  cudaSetDeviceFlags(cudaDeviceMapHost);


  unsigned long long zero = 0;
  uint blocks           = 64; 
  uint threads          = 128;
  size_t heapSize       = size_t(4U)*size_t(1024U*1024U*1024U);

  std::cout << "Gridsize:              " << blocks << std::endl;
  std::cout << "Blocksize:             " << threads << std::endl;
  std::cout << "Allocated elements:    " << sizeof(allocatedElement) << " Byte (" << 8*sizeof(allocatedElement) << " bit)" << std::endl;
  std::cout << "Heap:                  " << heapSize << " Byte (" << float(heapSize)/(1024*1024*1024) << " GByte)" << std::endl; 
  unsigned long long maxElements = heapSize/sizeof(allocatedElement);
  std::cout << "maximum of elements:   " << maxElements << std::endl;
  unsigned long long maxElementsP = heapSize/(sizeof(allocatedElement)+sizeof(allocatedElement*));
  std::cout << "maximum incl pointers: " << maxElementsP << std::endl;

// initializing
  initHeap(heapSize); 
  allocatedElement** d_testData;
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaHostAlloc((void**) &d_testData, (heapSize/sizeof(allocatedElement))*sizeof(allocatedElement*),cudaHostAllocMapped));
  //SCATTERALLOC_CUDA_CHECKED_CALL(cudaMalloc((void**) &d_testData, (heapSize/sizeof(allocatedElement))*sizeof(allocatedElement*)));


// allocating
  unsigned long long numberOfAllocatedElements = 0;
  unsigned long long sumOfAllocatedElements = 0;
  allocate(d_testData,&numberOfAllocatedElements,&sumOfAllocatedElements,blocks,threads);
  std::cout << "allocated elements:    " << numberOfAllocatedElements << "  (" << float(100*numberOfAllocatedElements)/maxElements<< "%)" << std::endl;
  std::cout << "wasted space:          " << ALLOCATION_SIZE * sizeof(allocatedElement)*(maxElements-numberOfAllocatedElements)/1024/1024 << " MByte" << std::endl;
  std::cout << "wasted incl pointers:  " << (sizeof(allocatedElement)+sizeof(allocatedElement*))*(maxElementsP-numberOfAllocatedElements)/1024/1024 << " MByte" << std::endl;

// verifying on device
  verify(d_testData,numberOfAllocatedElements);

// damaging
  std::cout << "\n\n\ndamaging of element...";
  CUDA_CHECK_KERNEL_SYNC(damageElement<<<1,1>>>(d_testData));
  std::cout << "done" << std::endl;

// verifying on device
  verify(d_testData,numberOfAllocatedElements);

  // release all memory
  std::cout << "deallocation...";
  unsigned long long* d_dealloc_counter;
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMalloc((void**) &d_dealloc_counter, sizeof(unsigned long long)));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMemcpy(d_dealloc_counter,&zero,sizeof(unsigned long long)*1,cudaMemcpyHostToDevice));
  CUDA_CHECK_KERNEL_SYNC(deAllocAll<<<blocks,threads>>>(d_testData,d_dealloc_counter,numberOfAllocatedElements));
  cudaFree(d_dealloc_counter);
  cudaFree(d_testData);
  std::cout << "done "<< std::endl;


}
