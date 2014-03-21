#pragma once

#include "policy_based_heap.cuh"

// global object
typedef GPUTools::PolicyAllocator< GPUTools::ScatteredHeap<SCATTERALLOC_HEAPARGS>, GPUTools::XMallocDistribution<SCATTERALLOC_HEAPARGS>, NullOnOOM, GPUTools::GetHeapSimpleMalloc > scatterAllocator_T;

__device__ scatterAllocator_T scatterAllocator;

// global initHeap
void* initHeap(size_t heapsize = 8U*1024U*1024U){
  void* pool;
  scatterAllocator_T* heap;
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaGetSymbolAddress((void**)&heap,scatterAllocator));
  SCATTERALLOC_CUDA_CHECKED_CALL(cudaMalloc(&pool, heapsize));
  GPUTools::initKernel<<<1,256>>>(heap,pool, heapsize);
  return pool;
};

#ifdef __CUDACC__
#if __CUDA_ARCH__ >= 200
// global overwrite malloc/free
__device__ void* malloc(size_t t) __THROW
{
  return scatterAllocator.alloc(t);
};
__device__ void  free(void* p) __THROW
{
  scatterAllocator.dealloc(p);
};
#endif
#endif

//TODO: globally overwrite new, new[], delete, delete[], placment new, placement new[]
