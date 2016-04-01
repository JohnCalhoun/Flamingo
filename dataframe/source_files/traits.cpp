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

#include "traits.inl"
template<class ... Type>
struct traits {
	typedef std::size_t			size_type; 
	static const size_type _numCol=sizeof...(Type);

	type_add<_numCol-1,Type...> type_add_recursive;
	size_type row_size(){ return type_add_recursive(); }; 

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

