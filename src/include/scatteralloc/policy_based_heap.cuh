#include "src/include/scatteralloc/utils.h"
#include "get_heap_simpleMalloc.cuh" /*GetHeapPolicy: GetHeapSimpleMalloc*/
#include "xmalloc_like_distribution.cuh" /*AllocationPolicy: XMallocDistribution*/
#include "null_on_oom_policy.cuh"    /*OOMPolicy: NullOnOOM*/
#include "scatterd_heap_policy.cuh"  /*CreationPolicy: ScatteredHeap */



// Host Class for Policies
namespace GPUTools{
  template < class CreationPolicy, class AllocationPolicy, class OOMPolicy, class GetHeapPolicy >
    class PolicyAllocator : public CreationPolicy, public AllocationPolicy, public OOMPolicy, public GetHeapPolicy
  {

    public:
      static const uint32 dataAlignment=0x10;

      __host__ void* init(void* memForHeap,size_t size){
        CreationPolicy::initK(memForHeap, size);
        return memForHeap;
      };

      __device__ void* alloc(size_t bytes){

        bytes = (bytes + dataAlignment - 1) & ~(dataAlignment-1); // TODO: own policy?
        uint32 req_size = AllocationPolicy::gather(bytes); //TODO: still needs pagesize

        void* myalloc   = CreationPolicy::create(req_size);
        const bool oom  = CreationPolicy::isOOM(myalloc);
        if(oom) myalloc = OOMPolicy::handleOOM(myalloc);

        //TODO: distribute executes similar code as gather. This is wasteful!
        void* myres     = AllocationPolicy::distribute(req_size,bytes,myalloc); //TODO: still needs pagesize

        return myres;
      };

      __device__ void dealloc(void* p){
        CreationPolicy::destroy(p);
      };

  };
}

