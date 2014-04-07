#include <stdio.h>

#ifdef __CUDACC__
#if __CUDA_ARCH__ >= 200
//__device__ void* malloc(size_t size) throw()
//{
//    printf("internal malloc\n");
//    return NULL ;
//}

//__device__ void* operator new(size_t size) throw(std::bad_alloc)
//{
//    printf("internal new\n");
//    return NULL ;
//}

__device__ void* operator new[](size_t size) throw(std::bad_alloc)
{
    printf("internal new[]\n\n");
    return NULL ;
}
#endif
#endif

__global__ void test(){
    printf("using malloc:\n");
   int* t = (int*) malloc(sizeof(int));

   printf("using new:\n");
   int* t2 = new int;

   printf("using new[]:\n");
   int* t3 = new int[32];

}

int main()
{
    printf("setting\n");
    cudaSetDevice(0);
    printf("start\n");
    test<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}
