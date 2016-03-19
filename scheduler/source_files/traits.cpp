//traits.cpp
#ifndef TRAITS_SCHEDULER_CPP
#define TRATIS_SCHEDULER_CPP
#include <tuple>
#include <tbb/flow_graph.h>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/placeholders.hpp>

namespace scheduler{

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

using boost::mpl::placeholders::_1;
template<class ... DataFrames>
struct traits {
	typedef boost::mpl::vector<DataFrames...>	type_vector;	
	typedef typename 
		transform<type_vector,std::add_lvalue_reference<_1> >::type reference_vector;
	typedef typename 
		vec2tuple<sizeof...(DataFrames)-1,reference_vector>::type		reference_tuple;
	
	typedef reference_tuple						Args;
	typedef typename std::function<void(Args&)>		Function; 
	typedef typename tbb::flow::continue_msg		Msg; 
	
};

}//scheduler
#endif
