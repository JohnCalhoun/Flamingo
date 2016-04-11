//traits
#ifndef TRAITS_VECTOR_CPP
#define TRAITS_VECTOR_CPP
#include <thrust/device_vector.h>
#include <thrust/host_vector.h> 

namespace Flamingo{
namespace Vector{ 

template<typename T,Memory M> 
struct Root_vector {
	typedef thrust::host_vector<T> type; 
};
template<typename T> 
struct Root_vector<T,device> {
	typedef thrust::device_vector<T> type; 
};

}
#endif
