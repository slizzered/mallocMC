project(mallocMC)
cmake_minimum_required(VERSION 3.0.1)

# helper for libs and packages
set(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/"
    "$ENV{CUDA_ROOT}" "$ENV{BOOST_ROOT}")


###############################################################################
# CUDA
###############################################################################
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "-use_fast_math;")
set(CUDA_ARCH sm_35 CACHE STRING "Set GPU architecture")
set(CUDA_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
cuda_include_directories(${CUDA_INCLUDE_DIRS})

OPTION(CUDA_OUTPUT_INTERMEDIATE_CODE "Output ptx code" OFF)
if(CUDA_OUTPUT_INTERMEDIATE_CODE)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xptxas;-v;--keep;-dc")
endif(CUDA_OUTPUT_INTERMEDIATE_CODE)

SET(CUDA_OPTIMIZATION_TYPE "-O3" CACHE STRING "CUDA Optimization")
set_property(CACHE CUDA_OPTIMIZATION_TYPE PROPERTY STRINGS "unset;-G0;-O0;-O1;-O2;-O3")
if(NOT ${CUDA_OPTIMIZATION_TYPE} STREQUAL  "unset")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};${CUDA_OPTIMIZATION_TYPE}")
endif()


###############################################################################
# Boost
###############################################################################
find_package(Boost 1.48.0 REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
set(LIBS ${LIBS} ${Boost_LIBRARIES})

# nvcc + boost 1.55 work around
if(Boost_VERSION EQUAL 105500)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} \"-DBOOST_NOINLINE=__attribute__((noinline))\" ")
endif(Boost_VERSION EQUAL 105500)


################################################################################
# Halloc
################################################################################
set(HALLOC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/halloc")
set(HALLOC_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/halloc")

set(CUDA_SEPARABLE_COMPILATION ON)
list(APPEND CUDA_NVCC_FLAGS "-rdc=true")

cuda_include_directories(${HALLOC_INSTALL_DIR}/include)
list(APPEND CUDA_NVCC_FLAGS "-lhalloc")
list(APPEND CUDA_NVCC_FLAGS "-L${HALLOC_INSTALL_DIR}/lib")
list(APPEND LIBS "${HALLOC_INSTALL_DIR}/lib/libhalloc.a")

add_custom_target(halloc
  COMMAND mkdir -p "${HALLOC_INSTALL_DIR}/include"
  COMMAND mkdir -p "${HALLOC_INSTALL_DIR}/lib"
  COMMAND make --directory=${HALLOC_DIR}
  COMMAND make install --directory=${HALLOC_DIR} PREFIX=${HALLOC_INSTALL_DIR}
  )


################################################################################
# Warnings
################################################################################
# GNU
if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
  # new warning in gcc 4.8 (flag ignored in previous version)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-local-typedefs")
  # ICC
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_VARIADIC_TEMPLATES")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_CXX11_VARIADIC_TEMPLATES")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_FENV_H")
  # PGI
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Minform=inform")
endif()


###############################################################################
# Installation
###############################################################################

# copy include folder to install folder
INSTALL(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/include/."
  DESTINATION include
  PATTERN ".git" EXCLUDE
  PATTERN "mallocMC_config.hpp" EXCLUDE
  )


###############################################################################
# Executables
###############################################################################
add_custom_target(examples DEPENDS mallocMC_Example01 mallocMC_Example02 mallocMC_Example03 VerifyHeap)

cuda_add_executable(mallocMC_Example01
                    EXCLUDE_FROM_ALL
                    examples/mallocMC_example01.cu )
cuda_add_executable(mallocMC_Example02
                    EXCLUDE_FROM_ALL
                    examples/mallocMC_example02.cu )
cuda_add_executable(mallocMC_Example03
                    EXCLUDE_FROM_ALL
                    examples/mallocMC_example03.cu )
add_dependencies(mallocMC_Example03 halloc)
cuda_add_executable(VerifyHeap
                    EXCLUDE_FROM_ALL
                    tests/verify_heap.cu )
add_dependencies(VerifyHeap halloc)


target_link_libraries(mallocMC_Example01 ${LIBS} )
target_link_libraries(mallocMC_Example02 ${LIBS} )
target_link_libraries(mallocMC_Example03 ${LIBS} )
target_link_libraries(VerifyHeap ${LIBS})
