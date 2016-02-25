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
#include <tuple>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <boost/type_traits/add_pointer.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/mpl/range_c.hpp>
#include <cstddef> 

template<typename vector>
struct vector2tuple{
	template<int n,typename vec, typename ... T>
	struct tuple_add {
		typedef typename boost::mpl::int_<n>				position;
		typedef typename boost::mpl::at<vec,position>::type	element;
		typedef typename tuple_add<n-1,vec,element>::type		type;
	};
	template<typename vec, typename ... T>
	struct tuple_add<0,vec,T...> {
		typedef std::tuple<T...>	type;
	};

	typedef typename boost::mpl::size<vector>::type Value;	
	typedef typename tuple_add<Value::value,vector>::type type;
};

using boost::mpl::placeholders::_1;
template<class ... Type>
struct traits {

	typedef boost::mpl::vector<Type...>	type_vector;	
	typedef typename boost::mpl::transform<type_vector,std::add_pointer<_1> >::type pointer_vector;
	typedef typename boost::mpl::transform<type_vector,std::add_lvalue_reference<_1> >::type reference_vector;

	typedef vector2tuple<type_vector>			value_tuple;
	typedef thrust::zip_iterator<value_tuple>	value_zip;

	typedef vector2tuple<pointer_vector>		pointer_tuple;
	typedef thrust::zip_iterator<pointer_tuple>	pointer_zip;

	typedef vector2tuple<reference_vector>			reference_tuple;
	typedef thrust::zip_iterator<reference_tuple>	reference_zip;

	typedef boost::mpl::range_c<int,0,sizeof...(Type)> range;

	typedef value_tuple			value_type;
	typedef pointer_tuple		pointer; 
	typedef reference_zip		reference;
	typedef std::size_t			size_type; 
	typedef std::ptrdiff_t		difference_type; 

	template<int n,Memory M>
	struct column_return{
		typedef boost::mpl::int_<n> value;

		typedef typename boost::mpl::at<type_vector,value >::type base;
		typedef typename column<base,M>::type type;  
	};

	template<int n>
	struct Return{
		typedef boost::mpl::int_<n> value;

		typedef typename boost::mpl::at<type_vector,value >::type		type_base;
		typedef typename boost::mpl::at<pointer_vector,value >::type	pointer_base;
		typedef typename boost::mpl::at<reference_vector,value >::type	reference_base;
	};
};

#endif 

