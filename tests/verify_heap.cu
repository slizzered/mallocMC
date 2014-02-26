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
#define SCATTERALLOC_HEAPARGS 4096, 8, 16, 2, true, false

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

  //SCATTERALLOC_CUDA_CHECKED_CALL(cudaDeviceSynchronize());
#define CUDA_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}
/*start kernel, wait for finish and check errors*/
#define CUDA_CHECK_KERNEL_SYNC(...) __VA_ARGS__;CUDA_CHECK(cudaDeviceSynchronize())
/*only check if kernel start is valid*/
#define CUDA_CHECK_KERNEL(...) __VA_ARGS__;CUDA_CHECK(cudaGetLastError())

typedef GPUTools::uint32 uint;

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
 * @brief allocate a lot of small arrays
 *
 *
 */
__global__ void allocAll(uint** array,uint elementsPerSlot){
  int gid = threadIdx.x + blockIdx.x*blockDim.x;
  array[gid] = new uint[elementsPerSlot];
  if(array[gid]==NULL){
    printf("Error: NULL)");
  }
}



/**
 * @brief fill all the values with a fixed value 
 *
 * one of those values will differ! (so you can check if the error-checker actually works)
 *
 */
__global__ void fillWithZero(uint** array,uint elementsPerSlot){
  int gid = threadIdx.x + blockIdx.x*blockDim.x;

  uint* data = array[gid];
  for(int i=0;i<elementsPerSlot;++i){
    data[i] = 274;
    if(gid==2 && i==3){
      data[i] = 273;
    }
  }
}


/**
 * @brief check if all the fields have the same value
 *
 */
__global__ void check_zerofilling(uint** array,uint elementsPerSlot){
  int gid = threadIdx.x + blockIdx.x*blockDim.x;

  uint* data = array[gid];

  for(int i=0;i<elementsPerSlot;++i){
    if(data[i] != 274){
      printf("Error in Kernel: TestData on position %d,%d is %d (should be %d)\n",gid,i,data[i],274);
    }
  }
}


/**
 * @brief fill all the fields with a unique ID
 *
 */
__global__ void fillWithIds(uint** array,uint elementsPerSlot){
  int gid = threadIdx.x + blockIdx.x*blockDim.x;

  uint* data = array[gid];
  for(int i=0;i<elementsPerSlot;++i){
    data[i] = uint(gid)*elementsPerSlot + uint(i);
    if(gid==2 && i==3){
      data[i] = data[i]-1;
    }
  }
}


/**
 * @brief checks on a per thread basis, if the gaussian sum is correct
 *
 */
__global__ void check_idfilling(uint** array,uint elementsPerSlot){
  int gid = threadIdx.x + blockIdx.x*blockDim.x;
  uint* data = array[gid];
  
  uint sum=0;
  for(int i=0;i<elementsPerSlot;++i){
    sum += (data[i] - uint(gid)*elementsPerSlot);
  }
  uint gaussian_sum = (elementsPerSlot * (elementsPerSlot-1))/2;
  if(gaussian_sum != sum){
    printf("Error: The sum over the testdata in thread %d is %d (should be %d)\n",gid,sum,gaussian_sum);
  }
}


/**
 * @brief free all the values again
 *
 */
__global__ void deAllocAll(uint** array){
  int gid = threadIdx.x + blockIdx.x*blockDim.x;
  delete[] array[gid];
}


/**
 * @brief verify that the heap actually holds the correct values without corrupting them
 *
 */
void run_heap_verification(int cuda_device)
{
  cudaSetDevice(cuda_device);
  std::cout << "start" << std::endl;

  size_t heapSize = size_t(3U)*uint(1024U*1024U*1024U); //can not exceed 4GB
  initHeap(heapSize); 
  std::cout << "heap initialized" << std::endl;

  uint blocks           = 32U*1024U; 
  uint threads          = 512U;
  uint slots            = blocks*threads;
  uint elementsPerSlot  = heapSize/slots/4;
  std::cout << "elementsPerSlot to fill the whole heap: "<<elementsPerSlot << std::endl;
  elementsPerSlot  = 40;
  std::cout << "elementsPerSlot actually used: "<<elementsPerSlot << std::endl;


  // create the datastructure on the device
  uint** dTestData;
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMalloc((void**) &dTestData, slots*sizeof(uint*)));
  CUDA_CHECK_KERNEL_SYNC(allocAll<<<blocks,threads>>>(dTestData,elementsPerSlot));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaDeviceSynchronize());
  std::cout << "allocation on device done" << std::endl;
  CUDA_CHECK_KERNEL_SYNC(checkIfRunning<<<1,1>>>());

  // initialize the datastructure on the device with a fixed value (and 1 error!)
  CUDA_CHECK_KERNEL_SYNC(fillWithZero<<<blocks,threads>>>(dTestData,elementsPerSlot));
  cudaDeviceSynchronize();
  std::cout << "zerofilling done" << std::endl;
  CUDA_CHECK_KERNEL_SYNC(checkIfRunning<<<1,1>>>());
  
  // validate the initialization directly on the device
  CUDA_CHECK_KERNEL_SYNC(check_zerofilling<<<blocks,threads>>>(dTestData,elementsPerSlot));
  cudaDeviceSynchronize();
  std::cout << "zerofilling checked" << std::endl;
  CUDA_CHECK_KERNEL_SYNC(checkIfRunning<<<1,1>>>());
  cudaDeviceSynchronize();

  // validate the initialization on the host
  //uint** hAddresses = (uint**) malloc(sizeof(uint*) * slots);
  //cudaMemcpy(hAddresses, (uint**) dTestData, slots * sizeof(uint*), cudaMemcpyDeviceToHost);
  //std::cout << "addresses copied" << std::endl;
  //for(unsigned long long i=0;i<slots;++i){
  //  uint* hData = (uint*) malloc(sizeof(uint) * elementsPerSlot);
  //  cudaMemcpy(hData, (uint*) hAddresses[i], elementsPerSlot * sizeof(uint), cudaMemcpyDeviceToHost);
  //  #pragma omp parallel for
  //  for(unsigned long long j=0; j<elementsPerSlot;++j){
  //    if(hData[j] != 274){
  //      std::cout << "Error: TestData on position " << i << "," << j  << " is " << hData[j] << " (should be 274)" << std::endl;
  //    }
  //  }
  //  free(hData);
  //}
  //free(hAddresses);
  //std::cerr << "verification of initialization done" << std::endl;
  //CUDA_CHECK_KERNEL_SYNC(checkIfRunning<<<1,1>>>());
  //cudaDeviceSynchronize();


  // fill the datastructure with unique IDs (ascending integers)
  fillWithIds<<<blocks,threads>>>(dTestData,elementsPerSlot);
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaDeviceSynchronize());
  std::cerr << "ID filling done" << std::endl;
  CUDA_CHECK_KERNEL_SYNC(checkIfRunning<<<1,1>>>());
  cudaDeviceSynchronize();

  CUDA_CHECK_KERNEL_SYNC(check_idfilling<<<blocks,threads>>>(dTestData,elementsPerSlot));
  cudaDeviceSynchronize();

  // verify the IDs using the gaussian sum formula
  //hAddresses = (uint**) malloc(sizeof(uint*) * slots);
  //cudaMemcpy(hAddresses, (uint**) dTestData, slots * sizeof(uint*), cudaMemcpyDeviceToHost);
  //unsigned long long sum =0;
  //for(unsigned i=0;i<slots;++i){
  //  uint* hData = (uint*) malloc(sizeof(uint) * elementsPerSlot);
  //  cudaMemcpy(hData, (uint*) hAddresses[i], elementsPerSlot * sizeof(uint), cudaMemcpyDeviceToHost);
  //  #pragma omp parallel for
  //  for(unsigned long long j=0;j<elementsPerSlot;++j){
  //    sum += hData[j];
  //  }
  //  free(hData);
  //}
  //free(hAddresses);
  //unsigned long long maxSumElement = (slots*elementsPerSlot)-1;
  //unsigned long long gaussian_sum = (maxSumElement * (maxSumElement+1))/2;
  //if(gaussian_sum != sum){
  //  std::cerr << "Error: The sum over the testdata is " << sum << " (should be " <<  gaussian_sum << ")" << std::endl;
  //}

  std::cout << "all IDs checked" << std::endl;

  // release all memory
  CUDA_CHECK_KERNEL_SYNC(deAllocAll<<<blocks,threads>>>(dTestData));
  cudaDeviceSynchronize();
  std::cerr << "deallocation done" << std::endl;
  CUDA_CHECK_KERNEL_SYNC(checkIfRunning<<<1,1>>>());
  cudaDeviceSynchronize();
  cudaFree(dTestData);
  std::cout << "end" << std::endl;

}
