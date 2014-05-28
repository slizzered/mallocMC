#include <iostream>
#include <stdio.h>


template <typename T>
__global__ void helperKernel(T* t)
{
  t->printSomething();
}


class TestClass{
  public:

    __device__ void printSomething()
    {
      printf("this works\n");
    }

    //This is the function that should be polymorphic over host/device
    __host__ __device__ void polymorphicCall()
    {
#ifdef __CUDA_ARCH__
      // on device, call the device function directly
      printf("calling from Device: ");
      printSomething();
#else
      // on host, start a helper kernel which will use the device function
      std::cerr << "calling from Host:   ";
      useHelperKernel(*this);
#endif
    }

  private:

    //This is the workaround. This function is NOT called
    __host__ void workaround()
    {
      useHelperKernel(*this); // code only works, when this line is present
    }

    template <typename T>
    void useHelperKernel(const T& a)
    {
      T* b;
      cudaGetSymbolAddress((void**)&b, a);
      helperKernel<<<1,1>>>(b);
    }
};



// a global instance
__device__ TestClass globalTestObject;


// calls code directly from inside a kernel
__global__ void callPolymorphicFromKernel()
{
  globalTestObject.polymorphicCall();
}


int main()
{
  callPolymorphicFromKernel<<<1,1>>>();
  cudaDeviceSynchronize();

  // run from the host. This will internally use a helper kernel
  globalTestObject.polymorphicCall();
  cudaDeviceSynchronize();

  std::cerr << "finished" << std::endl;
  return 0;
}
