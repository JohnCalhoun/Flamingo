//traits.cpp
#ifndef DATAFRAME_TRAITS
#define DATAFRAME_TRAITS

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
#include <boost/mpl/range_c.hpp>
#include <cstddef> 


template<int n,typename vec, typename ... T>
struct vec2tuple {
	typedef typename boost::mpl::int_<n>				position;
	typedef typename boost::mpl::at<vec,position>::type	element;
	typedef typename vec2tuple<n-1,vec,element,T...>::type		type;
};
template<typename vec, typename ... T>
struct vec2tuple<0,vec,T...> {
	typedef typename boost::mpl::int_<0>				position;
	typedef typename boost::mpl::at<vec,position>::type	element;

	typedef std::tuple<element,T...>	type;
};


template<typename vec, typename op>
struct transform{
	typedef typename boost::mpl::transform<vec,op>::type type; 
};	

template<typename T>
struct add_ref_wrap{
	typedef std::reference_wrapper<T> type; 
};

using boost::mpl::placeholders::_1;
template<class ... Type>
struct traits {
	typedef std::size_t			size_type; 
	static const size_type _numCol=sizeof...(Type);

	typedef boost::mpl::vector<Type...>	type_vector;	
	typedef typename transform<type_vector,std::add_pointer<_1> >::type pointer_vector;
	typedef typename transform<type_vector,std::add_lvalue_reference<_1> >::type reference_vector;

	typedef typename vec2tuple<_numCol-1,type_vector>::type		value_tuple;
	typedef typename vec2tuple<_numCol-1,pointer_vector>::type		pointer_tuple;
	typedef typename vec2tuple<_numCol-1,reference_vector>::type	reference_tuple;

	typedef boost::mpl::range_c<int,0,_numCol> range;

	typedef value_tuple			value_type;
	typedef pointer_tuple		pointer; 
	typedef reference_tuple		reference;
	typedef std::ptrdiff_t		difference_type; 
	template<int n>
	struct Return{
		typedef boost::mpl::int_<n> value;

		typedef typename 
			boost::mpl::at<type_vector,value >::type		type_base;
		typedef typename 
			boost::mpl::at<pointer_vector,value >::type		pointer_base;
		typedef typename 
			boost::mpl::at<reference_vector,value >::type	reference_base;
	};
};

#endif 

