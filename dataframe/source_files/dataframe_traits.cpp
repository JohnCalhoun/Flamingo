//traits.cpp
#ifndef DATAFRAME_DATAFRAME_TRAITS
#define DATAFRAME_DATAFRAME_TRAITS

#include <boost/mpl/vector.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/map.hpp>
#include <boost/type_traits/add_pointer.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/mpl/range_c.hpp>
#include <cstddef> 
#include <type_traits>
#include <functional>
#include "HashedArrayTree.cu"

namespace Flamingo{
namespace DataFrame{

template<class ... Type>
struct column_tuple {
	typedef traits<Type...> Traits; 

	template<typename U>
	struct type2column {
		typedef column<U> type; 
	};

	typedef typename Traits::type_vector vec; 
	typedef typename transform<vec,type2column<_1> >::type col_vec;
	typedef typename vec2tuple<Traits::_numCol-1,col_vec>::type type; 

	template<int n>
	struct element {
		typedef boost::mpl::int_<n> position;
		typedef typename boost::mpl::at<col_vec,position>::type type; 
	};
}; 

}//end dataframe
}//end flamingo
#endif 


