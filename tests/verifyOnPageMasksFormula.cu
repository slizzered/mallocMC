#include <iostream>
#include <stdint.h>
#include <vector>
#include <algorithm>
#include "progressbar.cu"
#include <stdio.h>

__device__ bool globalStatus = true;

__global__ void compareHTValue(const uint32_t pageSize, const uint32_t HierarchyThreshold, const uint32_t HTValue){
  int stride = blockDim.x*gridDim.x; 

  // 16 is the minimal ChunkSize in ScatterAlloc
  for(int chunkSize=16; chunkSize<=HierarchyThreshold; chunkSize=chunkSize+stride){
    const uint32_t segmentSize = chunkSize*32 + sizeof(uint32_t);
    const uint32_t fullSegments = pageSize / segmentSize;
    const uint32_t additionalChunks = max(0,pageSize - (int)fullSegments*segmentSize - (int)sizeof(uint32_t))/chunkSize;
    const uint32_t pointerPos = chunkSize*(32*fullSegments + additionalChunks);
    if(pointerPos < HTValue){
      globalStatus = false;
    }
  }
}

__global__ void checkFinalStatus(){
  if(globalStatus){
    printf("\nStatus: OK\n");
  }else{
    printf("\nStatus: FAIL\n");
  }
}

__global__ void checkStatusInbetween(){
  if(!globalStatus){
    printf("\nStatus: FAIL\n\n");
  }
}


int main(int argc, char* argv[]){

  if(argc < 3){
    std::cerr << "please submit pagesize limits!" << std::endl;
    return 1;
  }

  int pageSizeLow = atoi(argv[1]);
  int pageSizeHigh = atoi(argv[2]);


  for(int pageSize = pageSizeLow; pageSize <= pageSizeHigh; ++pageSize){
    const uint32_t HierarchyThreshold =  (pageSize - 2*sizeof(uint32_t))/33;

    // the newly found value of the minimal pointer position
    const uint32_t HTValue = 32*HierarchyThreshold;

    //determine all possible real values for the metadata pointer and check that HTValue is still below
    compareHTValue<<<128,64>>>(pageSize,HierarchyThreshold,HTValue);
    checkStatusInbetween<<<1,1>>>();
    cudaDeviceSynchronize();

    // progressbar
    fancyProgressBar(pageSizeHigh-pageSizeLow);
  }

  checkFinalStatus<<<1,1>>>();
  cudaDeviceSynchronize();

  return 0;
}
