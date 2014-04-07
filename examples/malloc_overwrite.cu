#include <stdio.h>

__device__ void* malloc(size_t size) throw()
{
    printf("Allocating memory with malloc\n");
    return NULL ;
}

__global__ void test(){
    printf("using malloc:\n");
    int* t = (int) malloc(sizeof(int));

    printf("using new:\n");
    int* t2 = new int;

    printf("using new[]:\n");
    int t3[] = new int[32];

}

int main()
{
    printf("setting\n");
    cudaSetDevice(0);
    printf("start\n");
    test<<<1,1>>>();
    printf("end\n");
    return 0;
}

