#pragma once

#include <cuda_runtime_api.h>

#include "CudaSetLimits.hpp"

namespace PolicyMalloc{
    
namespace GetHeapPolicies{

  struct CudaSetLimits{
      static void* getMemPool(size_t memsize){
        cudaDeviceSetLimit(cudaLimitMallocHeapSize, memsize);
        return NULL;
      }

    template < typename T>
      static void freeMemPool(const T& obj){
        //assert(!"freeMemPool is not implemented!");
        //TODO implement me!
      }
  };

} //namespace GetHeapPolicies

} //namespace PolicyMalloc
