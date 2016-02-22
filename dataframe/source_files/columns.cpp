//columns.cpp
#ifndef COLUMNS
#define COLUMNS
#include <allocator.cu>
#include <location.cu> 
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

class columnBase {};

template<typename T,Memory M>
struct column : public columnBase, public thrust::device_vector<T,typename allocation_policy<T,M>::allocator>{
	typedef thrust::device_vector<T,typename allocation_policy<T,M>::allocator> type;
	typedef location<M> location;
};

template<typename T>
struct column<T,host> : public columnBase, public thrust::host_vector<T,typename allocation_policy<T,host>::allocator> {
	typedef thrust::host_vector<T,typename allocation_policy<T,host>::allocator> type; 
	typedef location<host> location;
};

#endif 
