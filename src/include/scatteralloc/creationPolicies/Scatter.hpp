#pragma once
namespace PolicyMalloc{
namespace CreationPolicies{

  class PlaceHolder{};

  template<class T_Dummy=PlaceHolder>
  class Scatter2;

  typedef Scatter2<void> Scatter;

}// namespace CreationPolicies
}// namespace PolicyMalloc
