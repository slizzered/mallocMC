/*
  mallocMC: Memory Allocator for Many Core Architectures.
  https://www.hzdr.de/crp

  Copyright 2014 Institute of Radiation Physics,
                 Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Carlchristian Eckert - c.eckert ( at ) hzdr.de

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
*/

#include <iostream>
#include <assert.h>
#include <vector>
#include <numeric>

#include <cuda.h>
#include <boost/mpl/int.hpp>
#include <boost/mpl/bool.hpp>

///////////////////////////////////////////////////////////////////////////////
// includes for mallocMC
///////////////////////////////////////////////////////////////////////////////
// basic files for mallocMC
#include "src/include/mallocMC/mallocMC_overwrites.hpp"
#include "src/include/mallocMC/mallocMC_hostclass.hpp"

// Load all available policies for mallocMC
#include "src/include/mallocMC/CreationPolicies.hpp"
#include "src/include/mallocMC/DistributionPolicies.hpp"
#include "src/include/mallocMC/OOMPolicies.hpp"
#include "src/include/mallocMC/ReservePoolPolicies.hpp"
#include "src/include/mallocMC/AlignmentPolicies.hpp"
    
///////////////////////////////////////////////////////////////////////////////
// Configuration for mallocMC
///////////////////////////////////////////////////////////////////////////////

// configurate the CreationPolicy "Scatter"
struct ScatterConfig{
    typedef boost::mpl::int_<32*8192>  pagesize;
    typedef boost::mpl::int_<4>     accessblocks;
    typedef boost::mpl::int_<16>    regionsize;
    typedef boost::mpl::int_<2>     wastefactor;
    typedef boost::mpl::bool_<true> resetfreedpages;
};

struct ScatterHashParams{
    typedef boost::mpl::int_<38183> hashingK;
    typedef boost::mpl::int_<17497> hashingDistMP;
    typedef boost::mpl::int_<1>     hashingDistWP;
    typedef boost::mpl::int_<1>     hashingDistWPRel;
};

// configure the DistributionPolicy "XMallocSIMD"
struct DistributionConfig{
  typedef ScatterConfig::pagesize pagesize;
};

// configure the AlignmentPolicy "Shrink"
struct AlignmentConfig{
  typedef boost::mpl::int_<16> dataAlignment;
};

// Define a new allocator and call it ScatterAllocator
// which resembles the behaviour of ScatterAlloc
typedef mallocMC::Allocator< 
  mallocMC::CreationPolicies::Scatter<ScatterConfig,ScatterHashParams>,
  mallocMC::DistributionPolicies::XMallocSIMD<DistributionConfig>,
  mallocMC::OOMPolicies::ReturnNull,
  mallocMC::ReservePoolPolicies::SimpleCudaMalloc,
  mallocMC::AlignmentPolicies::Shrink<AlignmentConfig>
  > ScatterAllocator;

// use "ScatterAllocator" as mallocMC
MALLOCMC_SET_ALLOCATOR_TYPE(ScatterAllocator)

// replace all standard malloc()-calls on the device by mallocMC calls
// This will not work with the CreationPolicy "OldMalloc"!
MALLOCMC_OVERWRITE_MALLOC()

///////////////////////////////////////////////////////////////////////////////
// End of mallocMC configuration
///////////////////////////////////////////////////////////////////////////////


void run();

int main()
{
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  if( deviceProp.major < 2 ) {
    std::cerr << "Error: Compute Capability >= 2.0 required. (is ";
    std::cerr << deviceProp.major << "."<< deviceProp.minor << ")" << std::endl;
    return 1;
  }

  cudaSetDevice(0);
  run();
  cudaDeviceReset();

  return 0;
}


__global__ void fillBuffer(int** pagePointers, int* counter, const int chunkSize, const int maxChunks){
  id = threadIdx.x + blockIdx.x*blockDim.x;

  while(true){
    int pos = atomicAdd(counter,1);
    if(pos>=maxChunks){
      return;
    }

    int* p = (int*) malloc(chunkSize);
    if(p == NULL){
      //atomicSub(counter,1);
      return;
    }
    pagePointers[pos] = p;
  }
}



void run()
{
  size_t block = 32;
  size_t grid = 32;

  //init the heap
  std::cerr << "init...";
  size_t heapSize = 1U*1024U*1024U*1024U;
  int pageSize = ScatterConfig::pagesize::value;
  mallocMC::initHeap(heapSize); //1GB for device-side malloc. Yields 4096 usable pages
  // device-side pointers
  // this should be enough to hold exactly all possible pointers on mallocMC's heap
  int**  pagePointers;
  cudaMalloc((void**) &pagePointers, sizeof(int*)*heapSize/pageSize*1024); 

  thrust::host_vector<int> h_counter(1,0);
  thrust::device_vector<int> d_counter(h_counter);
  std::cerr << "done" << std::endl;
  int chunksPerPage = 32;
  size_t chunkSize = pageSize/chunksPerPage;
  int availableSlots=0;
  maxChunks = heapSize/pageSize*chunkSize;

  std::cout << "calculated Values:" << std::endl;
  std::cout << heapSize/pageSize << " pages of size " << pageSize << " byte available" << std::endl;
  std::cout << maxChunks << " slots of size " << chunkSize << " (" << chunksPerPage << " chunks per page)" << std::endl;
  
  availableSlots = mallocMC::getAvailableSlots(chunkSize);
  std::cout << "measured Values before filling:" << std::endl;
  std::cout << "availableSlots(" << chunkSize << ") = " << availableSlots << " (" << float(availableSlots)/maxChunks*100 << "%)"<< std::endl;

  fillBuffer<<<grid,block>>>(pagePointers,thrust::raw_pointer_cast(&d_counter[0]), chunkSize, maxChunks);
  std::count << "counter: " << d_counter[0] << std::endl;

  availableSlots = mallocMC::getAvailableSlots(chunkSize);
  std::cout << "measured Values after filling:" << std::endl;
  std::cout << "availableSlots(" << chunkSize << ") = " << availableSlots << " (" << float(availableSlots)/maxChunks*100 << "%)"<< std::endl;

//  clearBuffer<<<grid,block>>>(pagePointers,thrust::raw_pointer_cast(&d_counter[0]));

  availableSlots = mallocMC::getAvailableSlots(chunkSize);
  std::cout << "measured Values after freeing everything:" << std::endl;
  std::cout << "availableSlots(" << chunkSize << ") = " << availableSlots << " (" << float(availableSlots)/maxChunks*100 << "%)"<< std::endl;
  cudaFree(pagePointers);
  //finalize the heap again
  mallocMC::finalizeHeap();
}
