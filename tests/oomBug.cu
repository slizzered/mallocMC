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
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

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
    typedef boost::mpl::int_<4096>  pagesize;
    typedef boost::mpl::int_<8>     accessblocks;
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
  mallocMC::DistributionPolicies::Noop,
  mallocMC::OOMPolicies::ReturnNull,
  mallocMC::ReservePoolPolicies::SimpleCudaMalloc,
  mallocMC::AlignmentPolicies::Shrink<AlignmentConfig>
  > ScatterAllocator;

// use "ScatterAllocator" as mallocMC
MALLOCMC_SET_ALLOCATOR_TYPE(ScatterAllocator)

// replace all standard malloc()-calls on the device by mallocMC calls
// This will not work with the CreationPolicy "OldMalloc"!
//MALLOCMC_OVERWRITE_MALLOC()

///////////////////////////////////////////////////////////////////////////////
// End of mallocMC configuration
///////////////////////////////////////////////////////////////////////////////


void run(int chunksPerPage);

int main(int argc, char* argv[])
{
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  if( deviceProp.major < 2 ) {
    std::cerr << "Error: Compute Capability >= 2.0 required. (is ";
    std::cerr << deviceProp.major << "."<< deviceProp.minor << ")" << std::endl;
    return 1;
  }

  if(argc == 1){
    std::cerr << "Error: please supply the chunksize" << std::endl;
    return 1;
  }

  cudaSetDevice(0);
  run(atoi(argv[1]));
  cudaDeviceReset();

  return 0;
}


__global__ void fillBuffer(int* counter, const int chunkSize, const int maxChunks)
{
  while(*counter<maxChunks){
    int* p = (int*) mallocMC::malloc(chunkSize);
    if(p==NULL) return;
    atomicAdd(counter,1);
  }
}


void run(int chunkSize)
{
  size_t block = 128;
  size_t grid = 1;

  int availableSlots=0;
  int chunksPerPage = 0;
  int maxChunks = 0;
  int pageSize = ScatterConfig::pagesize::value;
  int HierarchyThreshold = (pageSize-2*sizeof(uint32_t))/33;
  size_t heapSize = 1U*4*1024U*1024U;
  thrust::device_vector<int> d_counter(1,0);

  // calculate some statistics for printing
  if(chunkSize <= HierarchyThreshold){
    int segmentSize = chunkSize*32 + sizeof(uint32_t);
    int fullSegments = pageSize / segmentSize;
    int additionalChunks = max(0,pageSize - fullSegments*segmentSize - (int)sizeof(uint32_t))/chunkSize;

    chunksPerPage = 32*fullSegments+additionalChunks;
    std::cout << "\n\033[0;31mUSING A HIERARCHICAL LAYOUT\033[0m with " << fullSegments << " segments and " << additionalChunks << " additional Chunks" << std::endl;
  }else{
    chunksPerPage=min(pageSize / chunkSize, 32);
  }
  uint32_t numregions = ((unsigned long long)heapSize)/( ((unsigned long long)ScatterConfig::regionsize::value)*(sizeof(uint32_t)*3+pageSize)+sizeof(uint32_t));
  uint32_t numpages = numregions*ScatterConfig::regionsize::value;


  //init the heap
  mallocMC::initHeap(heapSize);

  // print stuff before allocating
  availableSlots = mallocMC::getAvailableSlots(chunkSize);
  maxChunks = availableSlots;
  std::cout << "calculated Values:" << std::endl;
  std::cout << "at most: " << numpages << " pages of size " << pageSize << " byte available" << std::endl;
  std::cout << "(\033[0;32m" << chunksPerPage << "\033[0m chunks per page)" << std::endl;
  
  std::cout << "\nmeasured Values before filling:" << std::endl;
  std::cout << "availableSlots(" << chunkSize << ") = \033[0;33m" << availableSlots << "\033[0m (" << float(availableSlots)/maxChunks*100 << "% free)"<< std::endl;


  // allocate
  fillBuffer<<<grid,block>>>(thrust::raw_pointer_cast(&d_counter[0]), chunkSize, maxChunks);


  // print stuff after allocating
  availableSlots = mallocMC::getAvailableSlots(chunkSize);
  std::cout << "\nmeasured Values after filling:" << std::endl;
  std::cout << "availableSlots(" << chunkSize << ") = " << availableSlots << " (" << float(availableSlots)/maxChunks*100 << "% free)"<< std::endl;
  std::cout << "Successful allocations: \033[0;33m" << d_counter[0] << "\033[0m (" << (float(d_counter[0])/maxChunks)*100 << "% full)" << std::endl;

  //finalize the heap again
  mallocMC::finalizeHeap();
}
