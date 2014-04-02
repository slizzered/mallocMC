#pragma once

#include <boost/mpl/int.hpp>
#include <boost/mpl/bool.hpp>

#include "policy_malloc_hostclass.hpp"
#include "policy_malloc_overwrites.hpp"

#include "GetHeapPolicies.hpp"
#include "DistributionPolicies.hpp"
#include "OOMPolicies.hpp"
#include "CreationPolicies.hpp"

template<>
struct PolicyMalloc::GetProperties<PolicyMalloc::CreationPolicies::Scatter>{
    typedef boost::mpl::int_<4096>  pagesize;
    typedef boost::mpl::int_<8>     accessblocks;
    typedef boost::mpl::int_<16>    regionsize;
    typedef boost::mpl::int_<2>     wastefactor;
    typedef boost::mpl::bool_<false> resetfreedpages;
    typedef boost::mpl::int_<16>  dataAlignment;
};

struct DistributionTrait{
  typedef boost::mpl::int_<4096>  pagesize;
  typedef boost::mpl::int_<16>  dataAlignment;
};

typedef PolicyMalloc::PolicyAllocator< 
  PolicyMalloc::CreationPolicies::Scatter,
  PolicyMalloc::DistributionPolicies::XMallocSIMD<DistributionTrait>,
  PolicyMalloc::OOMPolicies::ReturnNull,
  PolicyMalloc::GetHeapPolicies::SimpleCudaMalloc
  > ScatterAllocator;


SET_ACCELERATOR_MEMORY_ALLOCATOR_TYPE(ScatterAllocator)
