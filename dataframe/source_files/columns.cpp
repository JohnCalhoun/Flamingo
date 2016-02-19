//columns.cpp
#ifndef COLUMNS
#define COLUMNS
#include <allocator.cu>
#include <location.cu> 
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template<typename T,typename L>
class column : public thrust::host_vector<T>{};

template<typename T>
class column<T,host> : public thrust::host_vector<
						T,
						typename allocation_policy<T,host>::allocator>{
};

template<typename T>
class column<T,device> : public thrust::device_vector<
						T,
						typename allocation_policy<T,device>::allocator>{
};
template<typename T>
class column<T,pinned> : public thrust::device_vector<
						T,
						typename allocation_policy<T,pinned>::allocator>{
};
template<typename T>
class column<T,unified> : public thrust::device_vector<
						T,
						typename allocation_policy<T,unified>::allocator>{
};

#endif 

