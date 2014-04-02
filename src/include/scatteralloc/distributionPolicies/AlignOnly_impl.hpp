#pragma once

#include <boost/cstdint.hpp>

#include "../policy_malloc_utils.hpp"
#include "AlignOnly.hpp"

namespace PolicyMalloc{
namespace DistributionPolicies{
    
    template <typename T_GetProperties>
    class AlignOnly 
    {
      typedef boost::uint32_t uint32;
      static const uint32 dataAlignment = T_GetProperties::dataAlignment::value;
      public:
        __device__ uint32 gather(uint32 bytes){
          bytes = (bytes + dataAlignment - 1) & ~(dataAlignment-1);
          return bytes;
        }

        __device__ void* distribute(void* allocatedMem){
          return allocatedMem;

        }

    };

} //namespace DistributionPolicies

} //namespace PolicyMalloc
