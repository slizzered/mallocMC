#pragma once

#include <boost/cstdint.hpp>

#include "../policy_malloc_utils.hpp"
#include "XMallocSIMD.hpp"

namespace PolicyMalloc{
namespace DistributionPolicies{

  template<class T_GetProperties>
    class XMallocSIMD
    {
      typedef boost::uint32_t uint32;
      bool can_use_coalescing;
      uint32 warpid;
      uint32 myoffset;
      uint32 threadcount;
      uint32 req_size;
      static const uint32 pagesize      = T_GetProperties::pagesize::value;
      static const uint32 dataAlignment = T_GetProperties::dataAlignment::value;




      public:
        __device__ uint32 gather(uint32 bytes){
          bytes = (bytes + dataAlignment - 1) & ~(dataAlignment-1);

          can_use_coalescing = false;
          warpid = PolicyMalloc::warpid();
          myoffset = 0;
          threadcount = 0;
        
          //init with initial counter
          __shared__ uint32 warp_sizecounter[32];
          warp_sizecounter[warpid] = 16;

          bool coalescible = bytes > 0 && bytes < (pagesize / 32);
          uint32 threadcount = __popc(__ballot(coalescible));

          if (coalescible && threadcount > 1) 
          {
            myoffset = atomicAdd(&warp_sizecounter[warpid], bytes);
            can_use_coalescing = true;
          }

          req_size = bytes;
          if (can_use_coalescing)
            req_size = (myoffset == 16) ? warp_sizecounter[warpid] : 0;

          return req_size;
        }

        __device__ void* distribute(void* allocatedMem){
          __shared__ char* warp_res[32];

          char* myalloc = (char*) allocatedMem;
          if (req_size && can_use_coalescing) 
          {
            warp_res[warpid] = myalloc;
            if (myalloc != 0)
              *(uint32*)myalloc = threadcount;
          }
          __threadfence_block();

          void *myres = myalloc;
          if(can_use_coalescing) 
          {
            if(warp_res[warpid] != 0)
              myres = warp_res[warpid] + myoffset;
            else 
              myres = 0;
          }
          return myres;

        }

    };

} //namespace DistributionPolicies

} //namespace PolicyMalloc
