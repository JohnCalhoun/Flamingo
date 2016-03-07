//functors.cpp
#include "columns.cpp"
#include "traits.cpp"

namespace iterator_functors{
	template<int n, class ... Type>
	struct assign{
		void operator()( 
				const typename traits<Type...>::ColumnArray& col_array,
				typename traits<Type...>::pointer& it_pointer)
		{
		typedef typename traits<Type...>::Return<n>::type_base type; 
		typedef typename traits<Type...>::Return<n>::pointer_base pointer; 
	

		column<type>* col_ptr=static_cast<column<type>*>(col_array[n]);
		if(col_ptr){
			void* void_ptr=col_ptr->access_raw(); 
			std::get<n>(it_pointer)=static_cast<pointer>(void_ptr);
		}else{
			std::get<n>(it_pointer)=NULL;
		}
		assign<n-1,Type...> assigner;
		assigner(col_array,it_pointer); 	
		}
	};
	template<class ... Type>
	struct assign<0,Type...>{
		void operator()(
				const typename traits<Type...>::ColumnArray& col_array,
				typename traits<Type...>::pointer& it_pointer)
		{
		typedef typename traits<Type...>::Return<0>::type_base type; 
		typedef typename traits<Type...>::Return<0>::pointer_base pointer; 
		
		column<type>* col_ptr=static_cast<column<type>*>(col_array[0]);
		if(col_ptr){
			void* void_ptr=col_ptr->access_raw(); 
			std::get<0>(it_pointer)=static_cast<pointer>(void_ptr);
		}else{
			std::get<0>(it_pointer)=NULL;
		}
		}
	};
}
