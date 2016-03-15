//functors.cpp
#include "columns.cpp"
#include "traits.cpp"

namespace iterator_functors{
	template<int n, class ... Type>
	struct assign{
		typedef typename column_tuple<Type...>::type				ColumnTuple; 
		typedef typename traits<Type...>::pointer				pointerTuple;
		typedef typename column_tuple<Type...>::element<n>::type	Column; 
		typedef typename traits<Type...>::Return<n>::pointer_base	pointer;

		void operator()( 
				ColumnTuple&&		columnTuple,
				pointerTuple&&		it_pointer)
		{
			Column& column=std::get<n>(columnTuple);
			void* void_ptr=column.access_raw(); 
			std::get<n>(it_pointer)=static_cast<pointer>(void_ptr);

			assign<n-1,Type...> assigner;
			assigner(	std::forward<ColumnTuple>(columnTuple),
					std::forward<pointerTuple>(it_pointer)); 	
		}
	};
	template<class ... Type>
	struct assign<0,Type...>{
		typedef typename column_tuple<Type...>::type				ColumnTuple; 
		typedef typename traits<Type...>::pointer				pointerTuple;
		typedef typename column_tuple<Type...>::element<0>::type	Column; 
		typedef typename traits<Type...>::Return<0>::pointer_base	pointer;

		void operator()( 
				ColumnTuple&&		columnTuple,
				pointerTuple&&		it_pointer)
		{
			Column& column=std::get<0>(columnTuple);
			void* void_ptr=column.access_raw(); 
			std::get<0>(it_pointer)=static_cast<pointer>(void_ptr);
		}
	};
	template<int n, class ... Type> 
	struct dereference{
		typedef typename traits<Type...>::pointer		pointer;
		typedef typename traits<Type...>::reference		reference; 
		typedef typename traits<Type...>::Return<n>::type_base	type;
		typedef type&			Element; 

		template<class ... ptr_types> 
		reference  operator()(	pointer&& ptr,
							ptr_types&&... args)
		{
			Element element=*std::get<n>(ptr);			
			dereference<n-1,Type...> recusive;


			return recusive(	std::forward<pointer>(ptr),
							std::forward<Element>(element),
							std::forward<ptr_types>(args)...
			); 
		}
		reference  operator()(	pointer&& ptr)
		{
			Element element=*std::get<n>(ptr);			

			dereference<n-1,Type...> recusive;


			return recusive(	std::forward<pointer>(ptr),
							std::forward<Element>(element)
			); 
		}
	};

	template<class ... Type> 
	struct dereference<0,Type...>{
		typedef typename traits<Type...>::pointer		pointer;
		typedef typename traits<Type...>::reference		reference; 
		typedef typename traits<Type...>::Return<0>::type_base	type;
		typedef type&			Element; 

		template<class ... ptr_types> 
		reference  operator()(	pointer&& ptr,
							ptr_types&&... args)
		{
			Element element=*std::get<0>(ptr);			

			return std::tie(	std::forward<Element>(element),
							std::forward<ptr_types>(args)...
			); 
		}
	};
}















