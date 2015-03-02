/*
  mallocMC: Memory Allocator for Many Core Architectures.
  http://www.icg.tugraz.at/project/mvp

  Copyright (C) 2015 Institute of Radiation Physics,
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

#include <boost/mpl/int.hpp>

namespace mallocMC{
namespace CreationPolicies{
namespace HallocConf{
  struct DefaultHallocConfig{
    typedef boost::mpl::int_<75>   halloc_fraction_nom;
    typedef boost::mpl::int_<100>  halloc_fraction_denom;
    typedef boost::mpl::int_<835>  busy_fraction_nom;
    typedef boost::mpl::int_<1000> busy_fraction_denom;
    typedef boost::mpl::int_<6>    roomy_fraction_nom;
    typedef boost::mpl::int_<10>   roomy_fraction_denom;
    typedef boost::mpl::int_<12>   sparse_fraction_nom;
    typedef boost::mpl::int_<1000> sparse_fraction_denom;
    typedef boost::mpl::int_<22>   sb_sz_sh;
  };

}

  /**
   * @brief 
   */
  template<class T_Config = HallocConf::DefaultHallocConfig>
  class Halloc;

}// namespace CreationPolicies
}// namespace mallocMC
