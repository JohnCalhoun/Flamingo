//traits.cpp
#ifndef DATAFRAME_TRAITS
#define DATAFRAME_TRAITS

#include "columns.cpp"
#include "iterator.cpp"
#include <boost/mpl/vector.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/transform.hpp>
#include <vector>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <boost/type_traits/add_pointer.hpp>
#include <boost/mpl/placeholders.hpp>
#include <cstddef> 

using boost::mpl::placeholders::_1;
template<class ... Type>
struct traits {

	typedef boost::mpl::vector<Type...>	type_vector;	
	typedef typename boost::mpl::transform<type_vector,std::add_pointer<_1> >::type pointer_vector;

	typedef thrust::tuple<type_vector>			value_tuple;
	typedef thrust::zip_iterator<value_tuple>	value_zip;

	typedef thrust::tuple<pointer_vector>		pointer_tuple;
	typedef thrust::zip_iterator<pointer_tuple>	pointer_zip;


	typedef value_tuple			value_type;
	typedef pointer_tuple		pointer; 
	typedef value_type&			reference;
	typedef std::size_t			size_type; 
	typedef std::ptrdiff_t		difference_type; 

	template<int n,Memory M>
	struct column_return{
		typedef typename boost::mpl::at<type_vector,boost::mpl::int_<n> >::type base;
		typedef typename column<base,M>::type type;  
	};
};

#endif 

