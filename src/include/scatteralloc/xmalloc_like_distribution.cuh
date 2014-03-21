#include "src/include/scatteralloc/utils.h"
namespace GPUTools{

  template<uint32 pagesize = 4096, uint32 accessblocks = 8, uint32 regionsize = 16, uint32 wastefactor = 2, bool use_coalescing = true, bool resetfreedpages = false>
    class XMallocDistribution
    {
      public:
        __device__ uint32 gather(uint32 bytes){
          //shared structs to use
          __shared__ uint32 warp_sizecounter[32];

          bool can_use_coalescing = false; 
          uint32 myoffset = 0;
          uint32 warpid = GPUTools::warpid();

          //init with initial counter
          warp_sizecounter[warpid] = 16;

          bool coalescible = bytes > 0 && bytes < (pagesize / 32);
          uint32 threadcount = __popc(__ballot(coalescible));

          if (coalescible && threadcount > 1) 
          {
            myoffset = atomicAdd(&warp_sizecounter[warpid], bytes);
            can_use_coalescing = true;
          }

          uint32 req_size = bytes;
          if (can_use_coalescing)
            req_size = (myoffset == 16) ? warp_sizecounter[warpid] : 0;

          return req_size;
        }

        __device__ void* distribute(uint32 req_size, uint32 bytes,void* allocatedMem){
          //shared structs to use
          __shared__ uint32 warp_sizecounter[32];

          bool can_use_coalescing = false; 
          uint32 myoffset = 0;
          uint32 warpid = GPUTools::warpid();

          //init with initial counter
          warp_sizecounter[warpid] = 16;

          bool coalescible = bytes > 0 && bytes < (pagesize / 32);
          uint32 threadcount = __popc(__ballot(coalescible));

          if (coalescible && threadcount > 1) 
          {
            myoffset = atomicAdd(&warp_sizecounter[warpid], bytes);
            can_use_coalescing = true;
          }

          // up to here, the code is basically identical with "gather"
          // (however, many of the created variables are needed from now on

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
}
