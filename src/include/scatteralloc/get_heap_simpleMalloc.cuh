#include "src/include/scatteralloc/utils.h"

namespace GPUTools{
  class GetHeapSimpleMalloc
  {
    public:
      static void* getHeapMem(size_t memsize){
        void* pool;
        SCATTERALLOC_CUDA_CHECKED_CALL(cudaMalloc(&pool, memsize));
        return pool;
      }
  };
}
