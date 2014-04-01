#pragma once

#define SET_ACCELERATOR_MEMORY_ALLOCATOR_TYPE(POLICY_MALLOC_USER_DEFINED_TYPENAME)           \
typedef POLICY_MALLOC_USER_DEFINED_TYPENAME PolicyMallocType;                  \
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
                                                                               \
__host__ void destroyHeap(PolicyMallocType &p = policyMallocGlobalObject)      \
{                                                                              \
  return PolicyMallocType::destroyHeap(p);                                     \
}                                                                              \
                                                                               \
__device__ void* malloc(size_t t) __THROW                                      \
{                                                                              \
  return policyMallocGlobalObject.alloc(t);                                    \
}                                                                              \
                                                                               \
__device__ void  free(void* p) __THROW                                         \
{                                                                              \
  policyMallocGlobalObject.dealloc(p);                                         \
}                                                                              \
                                                                               \

//TODO: globally overwrite new, new[], delete, delete[], placment new, placement new[]
