//columns.cpp
#ifndef COLUMNS
#define COLUMNS
#include <allocator.cu>
#include <location.cu> 
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template<typename T,Memory M>
struct column_traits {
	typedef thrust::device_vector<T,typename allocation_policy<T,M>::allocator> column;
	typedef location<M> location;
};

template<typename T>
struct column_traits<T,host> {
	typedef thrust::host_vector<T,typename allocation_policy<T,host>::allocator> column;
	typedef location<host> location;
};

#endif 
