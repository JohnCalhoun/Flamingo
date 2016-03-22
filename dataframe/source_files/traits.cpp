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
#include <type_traits>
#include <functional>

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

template<int n,typename vec>
struct assert_pod {
	typedef typename boost::mpl::int_<n>				position;
	typedef typename boost::mpl::at<vec,position>::type	element;
	
	typedef typename std::conditional<	
		std::is_pod<element>::value,
		typename assert_pod<n-1,vec>::type,
		typename std::false_type::type
						>::type type; 
};
template<typename vec>
struct assert_pod<0,vec> {
	typedef typename boost::mpl::int_<0>				position;
	typedef typename boost::mpl::at<vec,position>::type	element;
	
	typedef typename std::conditional<	
		std::is_pod<element>::value,
		typename std::true_type,
		typename std::false_type
						>::type type; 
};
template<typename T>
struct add_ptr_to_const{
	typedef const T* type; 
};

template<typename T>
struct add_const_ref{
	typedef std::reference_wrapper<const T> type; 
};

using boost::mpl::placeholders::_1;
template<class ... Type>
struct traits {
	typedef std::size_t			size_type; 
	static const size_type _numCol=sizeof...(Type);

	typedef boost::mpl::vector<Type...>	type_vector;	
	static_assert(assert_pod<_numCol-1,type_vector>::type::value,"DataFrame Types must be POD"); 

	typedef typename 
		transform<type_vector,std::add_pointer<_1> >::type		pointer_vector;
	typedef typename 
		transform<type_vector,std::add_lvalue_reference<_1> >::type reference_vector;
	typedef typename 
		transform<	type_vector,
					add_ptr_to_const<_1> 
				>::type	const_pointer_vector;
	typedef typename 
		transform<type_vector,add_const_ref<_1> >::type		const_reference_vector;

	typedef typename 
		vec2tuple<_numCol-1,type_vector>::type			value_tuple;
	typedef typename 
		vec2tuple<_numCol-1,pointer_vector>::type		pointer_tuple;
	typedef typename 
		vec2tuple<_numCol-1,reference_vector>::type		reference_tuple;
	typedef typename 
		vec2tuple<_numCol-1,const_pointer_vector>::type	const_pointer_tuple;
	typedef typename 
		vec2tuple<_numCol-1,const_reference_vector>::type	const_reference_tuple;

	typedef boost::mpl::range_c<int,0,_numCol> range;

	typedef value_tuple			value_type;
	typedef pointer_tuple		pointer; 
	typedef const_pointer_tuple	const_pointer;
	typedef reference_tuple		reference;
	typedef const_reference_tuple	const_reference;

	typedef thrust::zip_iterator<pointer>		zip_iterator; 
	typedef thrust::zip_iterator<const_pointer>	const_zip_iterator; 
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

