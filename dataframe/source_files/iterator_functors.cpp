//functors.cpp
#include "columns.cpp"
#include "traits.cpp"

namespace iterator_functors{
	template<int n, class ... Type>
	struct assign{
		void operator()( 
				typename column_tuple<Type...>::type& columnTuple,
				typename traits<Type...>::pointer& it_pointer)
		{
		typedef typename column_tuple<Type...>::type			col_tup; 
		typedef typename column_tuple<Type...>::element<n>::type	type; 
		typedef typename traits<Type...>::Return<n>::pointer_base	pointer;

		type& column=std::get<n>(columnTuple);
		void* void_ptr=column.access_raw(); 
		std::get<n>(it_pointer)=static_cast<pointer>(void_ptr);

		assign<n-1,Type...> assigner;
		assigner(columnTuple,it_pointer); 	
		}
	};
	template<class ... Type>
	struct assign<0,Type...>{
		void operator()(
				typename column_tuple<Type...>::type& columnTuple,
				typename traits<Type...>::pointer& it_pointer)
		{
		typedef typename column_tuple<Type...>::type			col_tup; 
		typedef typename column_tuple<Type...>::element<0>::type			type; 
		typedef typename traits<Type...>::Return<0>::pointer_base	pointer;

		type& column=std::get<0>(columnTuple);
		void* void_ptr=column.access_raw(); 
		std::get<0>(it_pointer)=static_cast<pointer>(void_ptr);	
		}
	};
}
