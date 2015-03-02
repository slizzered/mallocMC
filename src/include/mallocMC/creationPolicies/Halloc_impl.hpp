/*
  mallocMC: Memory Allocator for Many Core Architectures.

  Copyright 2015 Institute of Radiation Physics,
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

#pragma once

#include <boost/cstdint.hpp>
#include <boost/mpl/bool.hpp>

#include "Halloc.hpp"
#include "halloc.h"

namespace mallocMC{
namespace CreationPolicies{
    
  template<class T_Config>
  class Halloc
  {
    typedef boost::uint32_t uint32;

    public:
    typedef T_Config Properties;
    typedef boost::mpl::bool_<false> providesAvailableSlots;

    __device__ void* create(uint32 bytes)
    {
      return hamalloc(static_cast<size_t>(bytes));
    }

    __device__ void destroy(void* mem)
    {
      ::hafree(mem);
    }

    __device__ bool isOOM(void* p, size_t s){
      return s && (p == NULL);
    }

    template <typename T>
    static void* initHeap(const T& obj, void* pool, size_t memsize){
/** Allow for a hierarchical validation of parameters:
 *
 * shipped default-parameters (in the inherited struct) have lowest precedence.
 * They will be overridden by a given configuration struct. However, even the
 * given configuration struct can be overridden by compile-time command line
 * parameters (e.g. -D MALLOCMC_CP_HALLOC_HALLOCFRACTION 0.75)
 *
 * default-struct < template-struct < command-line parameter
 */
#ifndef MALLOCMC_CP_HALLOC_HALLOCFRACTION
#define MALLOCMC_CP_HALLOC_HALLOCFRACTION \
      static_cast<double>(Properties::halloc_fraction_nom::value) \
      / static_cast<double>(Properties::halloc_fraction_denom::value)
#endif

#ifndef MALLOCMC_CP_HALLOC_BUSYFRACTION
#define MALLOCMC_CP_HALLOC_BUSYFRACTION \
      static_cast<double>(Properties::busy_fraction_nom::value) \
      / static_cast<double>(Properties::busy_fraction_denom::value)
#endif

#ifndef MALLOCMC_CP_HALLOC_ROOMYFRACTION
#define MALLOCMC_CP_HALLOC_ROOMYFRACTION \
      static_cast<double>(Properties::roomy_fraction_nom::value) \
      / static_cast<double>(Properties::roomy_fraction_denom::value)
#endif

#ifndef MALLOCMC_CP_HALLOC_SPARSEFRACTION
#define MALLOCMC_CP_HALLOC_SPARSEFRACTION \
      static_cast<double>(Properties::sparse_fraction_nom::value) \
      / static_cast<double>(Properties::sparse_fraction_denom::value)
#endif

#ifndef MALLOCMC_CP_HALLOC_SBSZSH
#define MALLOCMC_CP_HALLOC_SBSZSH static_cast<int>(Properties::sb_sz_sh::value)
#endif

      halloc_opts_t default_opts(memsize);
      default_opts.halloc_fraction = MALLOCMC_CP_HALLOC_HALLOCFRACTION;
      default_opts.busy_fraction   = MALLOCMC_CP_HALLOC_BUSYFRACTION;
      default_opts.roomy_fraction  = MALLOCMC_CP_HALLOC_ROOMYFRACTION;
      default_opts.sparse_fraction = MALLOCMC_CP_HALLOC_SPARSEFRACTION;
      default_opts.sb_sz_sh        = MALLOCMC_CP_HALLOC_SBSZSH;
      ::ha_init(default_opts);
      return NULL;
    }

    template <typename T>
    static void finalizeHeap(const T& obj, void* pool){
      ::ha_shutdown();
    }

    static std::string classname(){
      return "Halloc";
    }

  };

} //namespace CreationPolicies
} //namespace mallocMC
