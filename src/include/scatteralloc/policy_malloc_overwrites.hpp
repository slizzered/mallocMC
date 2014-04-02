#pragma once

#define POLICY_MALLOC_GLOBAL_FUNCTIONS_INTERNAL(POLICY_MALLOC_USER_DEFINED_TYPENAME_INTERNAL)           \
typedef POLICY_MALLOC_USER_DEFINED_TYPENAME_INTERNAL PolicyMallocType;         \
                                                                               \
__device__ PolicyMallocType policyMallocGlobalObject;                          \
                                                                               \
__host__ static void* initHeap(                                                \
    size_t heapsize = 8U*1024U*1024U,                                          \
    PolicyMallocType &p = policyMallocGlobalObject                             \
    )                                                                          \
{                                                                              \
  return PolicyMallocType::initHeap(p,heapsize);                               \
}                                                                              \
__host__ void destroyHeap(PolicyMallocType &p = policyMallocGlobalObject)      \
{                                                                              \
  return PolicyMallocType::destroyHeap(p);                                     \
}                                                                              
                                                                               

#ifdef __CUDACC__
#if __CUDA_ARCH__ >= 200
#define POLICY_MALLOC_MEMORY_ALLOCATOR_MALLOC_OVERWRITE()                      \
                                                                               \
__device__ void* malloc(size_t t) __THROW                                      \
{                                                                              \
  return policyMallocGlobalObject.alloc(t);                                    \
}                                                                              \
                                                                               \
__device__ void  free(void* p) __THROW                                         \
{                                                                              \
  policyMallocGlobalObject.dealloc(p);                                         \
}                                                                              
#else
#define POLICY_MALLOC_MEMORY_ALLOCATOR_MALLOC_OVERWRITE()
#endif
#endif

#define SET_ACCELERATOR_MEMORY_ALLOCATOR_TYPE(POLICY_MALLOC_USER_DEFINED_TYPE)\
POLICY_MALLOC_GLOBAL_FUNCTIONS_INTERNAL(POLICY_MALLOC_USER_DEFINED_TYPE)\
POLICY_MALLOC_MEMORY_ALLOCATOR_MALLOC_OVERWRITE()


//TODO: globally overwrite new, new[], delete, delete[], placment new, placement new[]
