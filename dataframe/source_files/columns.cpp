//columns.cpp
#ifndef COLUMNS
#define COLUMNS
#include <allocator.cu>
#include <location.cu> 
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template<typename T,typename L>
struct column_traits {
	typedef thrust::device_vector<T,typename allocation_policy<T,L>::allocator> column;
	typedef location<L> location;
};

template<typename T>
struct column_traits<T,host> {
	typedef thrust::host_vector<T,typename allocation_policy<T,host>::allocator> column;
	typedef location<host> location;
};

#endif 
