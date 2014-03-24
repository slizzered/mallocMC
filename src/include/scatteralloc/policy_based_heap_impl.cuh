#pragma once

#include "policy_based_heap.cuh"
#include "get_heap_simpleMalloc.cuh" /*GetHeapPolicy: GetHeapSimpleMalloc*/
#include "xmalloc_like_distribution.cuh" /*AllocationPolicy: XMallocDistribution*/
#include "null_on_oom_policy.cuh"    /*OOMPolicy: NullOnOOM*/
#include "scatterd_heap_policy.cuh"  /*CreationPolicy: ScatteredHeap */



// global object
typedef GPUTools::PolicyAllocator< GPUTools::ScatteredHeap<SCATTERALLOC_HEAPARGS>, GPUTools::XMallocDistribution<SCATTERALLOC_HEAPARGS>, NullOnOOM, GPUTools::GetHeapSimpleMalloc > ScatterAllocator;


//typedef OtherAllocator PolClass;
typedef  ScatterAllocator PolClass;


__device__ PolClass polObject;

// global initHeap
__host__ void* initHeap(size_t heapsize = 8U*1024U*1024U){
  return PolClass::initHeap(polObject,heapsize);
};

__host__ void* initHeap(PolClass p,size_t heapsize = 8U*1024U*1024U){
  return PolClass::initHeap(p,heapsize);
};

__host__ void destroyHeap(){
  PolClass::destroyHeap(polObject);
};

__host__ void destroyHeap(PolClass p){
  PolClass::destroyHeap(p);
};

#ifdef __CUDACC__
#if __CUDA_ARCH__ >= 200
// global overwrite malloc/free
__device__ void* malloc(size_t t) __THROW
{
  return polObject.alloc(t);
};
__device__ void  free(void* p) __THROW
{
  polObject.dealloc(p);
};
#endif
#endif

//TODO: globally overwrite new, new[], delete, delete[], placment new, placement new[]
