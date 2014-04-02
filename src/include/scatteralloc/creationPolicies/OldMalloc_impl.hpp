#pragma once

#include <boost/cstdint.hpp>

#include "OldMalloc.hpp"


namespace PolicyMalloc{
namespace CreationPolicies{
    


  //__device__ static void* (*old_malloc)(size_t);
  //__device__ static void (*old_free)(void*);
  //
  //__global__ void save_malloc_addresses(){
  //     old_malloc = dlsym(RTLD_NEXT, "malloc");
  //     old_free   = dlsym(RTLD_NEXT, "free");
  //}
  
  __device__ static void* (*old_malloc)(size_t) = malloc;
  __device__ static void (*old_free)(void*)     = free;



  class OldMalloc
  {
    typedef boost::uint32_t uint32;

    public:
    __device__ void* create(uint32 bytes)
    {
      return old_malloc(static_cast<size_t>(bytes));
    }

    __device__ void destroy(void* mem)
    {
      old_free(mem);
    }

    __device__ bool isOOM(void* p){
      return  32 == __popc(__ballot(p == NULL));
    }


    template < typename T>
    static void* initHeap(const T& obj, void*pool, size_t memsize){
      //save_malloc_addresses<<<1,1>>>();
      return NULL;
    }   

  };


} //namespace CreationPolicies

} //namespace PolicyMalloc
