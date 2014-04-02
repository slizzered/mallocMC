#pragma once

#include "../policy_malloc_utils.hpp"
#include "SimpleCudaMalloc.hpp"

namespace PolicyMalloc{
    
namespace GetHeapPolicies{

  struct SimpleCudaMalloc{
      static void* getMemPool(size_t memsize){
        void* pool;
        SCATTERALLOC_CUDA_CHECKED_CALL(cudaMalloc(&pool, memsize));
        return pool;
      }

    template < typename T>
      static void freeMemPool(const T& obj){
        //assert(!"freeMemPool is not implemented!");
        //TODO implement me!
      }
  };

} //namespace GetHeapPolicies

} //namespace PolicyMalloc
