#pragma once 

#include "policy_malloc_utils.hpp"
#include <boost/cstdint.hpp>


namespace PolicyMalloc{

  template <typename T>
  struct GetProperties;

  class PlaceHolder{};

  template < typename T_Allocator >
    __global__ void initKernel(T_Allocator* heap, void* heapmem, size_t memsize){
      heap->initDeviceFunction(heapmem, memsize);
    }


  template < 
     typename T_CreationPolicy, 
     typename T_DistributionPolicy, 
     typename T_OOMPolicy, 
     typename T_GetHeapPolicy
       >
  struct PolicyAllocator : 
    public T_CreationPolicy, 
    public T_DistributionPolicy, 
    public T_OOMPolicy, 
    public T_GetHeapPolicy
  {
    private:
      typedef boost::uint32_t uint32;
      typedef T_CreationPolicy CreationPolicy;
      typedef T_DistributionPolicy DistributionPolicy;
      typedef T_OOMPolicy OOMPolicy;
      typedef T_GetHeapPolicy GetHeapPolicy;

    public:
      typedef PolicyAllocator<CreationPolicy,DistributionPolicy,OOMPolicy,GetHeapPolicy> MyType;
      __device__ void* alloc(size_t bytes){
        DistributionPolicy distributionPolicy;

        uint32 req_size  = distributionPolicy.gather(bytes);
        void* memBlock   = CreationPolicy::create(req_size);
        const bool oom   = CreationPolicy::isOOM(memBlock);
        if(oom) memBlock = OOMPolicy::handleOOM(memBlock);
        void* myPart     = distributionPolicy.distribute(memBlock);

        return myPart;
        // if(blockIdx.x==0 && threadIdx.x==0){
        //     printf("warp %d trying to allocate %d bytes. myalloc: %p (oom %d)\n",GPUTools::warpid(),req_size,myalloc,oom);
        // }
      }

      __device__ void dealloc(void* p){
        CreationPolicy::destroy(p);
      }

      __host__ static void* initHeap(const MyType& obj, size_t size){
        void* pool = GetHeapPolicy::getMemPool(size);
        return CreationPolicy::initHeap(obj,pool,size);
      }

      __host__ static void destroyHeap(const MyType& obj){
        GetHeapPolicy::freeMemPool(obj);
      }

  };

} //namespace PolicyMalloc
