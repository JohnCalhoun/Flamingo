//columns.cpp
#ifndef COLUMNS
#define COLUMNS
#include <allocator.cu>
#include <location.cu> 
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <boost/mpl/int.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/vector.hpp>

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

template<int n,typename L,typename vec>
struct column_return{
	typedef typename boost::mpl::at<vec,boost::mpl::int_<n> >::type base;
	typedef typename column_traits<base,L>::column* type;  
};

#endif 
