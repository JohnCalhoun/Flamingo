//traits.inl
#ifndef DATAFRAME_TRAITS_INL
#define DATAFRAME_TRAITS_INL

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

template<int n, typename current,class ... Type>
struct type_add {
	

	size_t operator()(){
		size_t tmp=sizeof(current); 

		type_add<n-1,Type...> recursive;
		return tmp+recursive(); 
	}
};
template<typename current,class ... Type>
struct type_add<0,current,Type...> {
	size_t operator()(){
		return sizeof(current); 
	}
};


using boost::mpl::placeholders::_1;
#endif 















