//functors.cpp
#include "columns.cpp"
#include "traits.cpp"
#include "functional"
#include "iterator.cpp"

namespace iterator_functors{
	template<int n, class ... Type>
	struct assign_const{
		typedef typename column_tuple<Type...>::type				ColumnTuple; 
		typedef typename traits<Type...>::const_pointer			pointerTuple;
		typedef typename column_tuple<Type...>::element<n>::type	Column; 

		void operator()( 
				const ColumnTuple&&		columnTuple,
				pointerTuple&&			it_pointer)
		{
			const Column& column=std::get<n>(columnTuple);
			std::get<n>(it_pointer)=column.data();

			assign_const<n-1,Type...> assigner;
			assigner(	std::forward<const ColumnTuple>(columnTuple),
					std::forward<pointerTuple>(it_pointer)); 	
		}
	};
	template<class ... Type>
	struct assign_const<0,Type...>{
		typedef typename column_tuple<Type...>::type				ColumnTuple; 
		typedef typename traits<Type...>::const_pointer			pointerTuple;
		typedef typename column_tuple<Type...>::element<0>::type	Column; 

		void operator()( 
				const ColumnTuple&&		columnTuple,
				pointerTuple&&			it_pointer)
		{
			const Column& column=std::get<0>(columnTuple);
			std::get<0>(it_pointer)=column.data();
		}
	};
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
			std::get<n>(it_pointer)=static_cast<pointer>(column.data());

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
			std::get<0>(it_pointer)=static_cast<pointer>(column.data());
		}
	};
	template<int n ,typename Ref,typename Ptr, class ... Type> 
	struct dereference{
		typedef Ptr		pointer;
		typedef Ref		reference; 
		typedef typename std::tuple_element<n,reference>::type	Element;

		template<class ... ptr_types> 
		reference  operator()(	pointer&& ptr,
							ptr_types&&... args)
		{
			Element element=*std::get<n>(ptr);			
			dereference<n-1,Ref,Ptr,Type...> recusive;

			return recusive(	std::forward<pointer>(ptr),
							std::forward<Element>(element),
							std::forward<ptr_types>(args)...
			); 
		}
		reference  operator()(	pointer&& ptr)
		{
			Element element=*std::get<n>(ptr);			

			dereference<n-1,Ref,Ptr,Type...> recusive;


			return recusive(	std::forward<pointer>(ptr),
							std::forward<Element>(element)
			); 
		}
	};

	template<typename Ref,typename Ptr,class ... Type> 
	struct dereference<0,Ref,Ptr,Type...>{
		typedef Ptr		pointer;
		typedef Ref		reference; 
		typedef typename std::tuple_element<0,reference>::type	Element;

		template<class ... ptr_types> 
		reference  operator()(	pointer&& ptr,
							ptr_types&&... args)
		{
			Element element=*std::get<0>(ptr);			

			return reference(	std::forward<Element>(element),
							std::forward<ptr_types>(args)...
			); 
		}
	};
	template<int n,typename pointer>
	struct nullify {
		void operator()(pointer&& p){
			std::get<n>(p)=NULL;		
			nullify<n-1,pointer> null_r;
			null_r(std::forward<pointer>(p));	
		}
	};
	template<typename pointer>
	struct nullify<0,pointer> {
		void operator()(pointer&& p){
			std::get<0>(p)=NULL;	
		}
	};
	//increment
	template<int n,typename pointer>
	struct increment {
		void operator()(pointer&& p){

			auto tmp=std::get<n>(p);
			tmp++;		
			std::get<n>(p)=tmp;					
			
			increment<n-1,pointer> inc_r;
			inc_r(std::forward<pointer>(p));	
		}
	};
	template<typename pointer>
	struct increment<0,pointer> {
		void operator()(pointer&& p){	
			auto tmp=std::get<0>(p);
			tmp++;		
			std::get<0>(p)=tmp;				
		}
	};
	//decrement
	template<int n,typename pointer>
	struct decrement {
		void operator()(pointer&& p){

			pointer tmp=std::get<n>(p);
			tmp--;		
			std::get<n>(p)=tmp;					
			
			decrement<n-1,pointer> inc_r;
			inc_r(std::forward<pointer>(p));	
		}
	};
	template<typename pointer>
	struct decrement<0,pointer> {
		void operator()(pointer&& p){	
			pointer tmp=std::get<0>(p);
			tmp--;		
			std::get<0>(p)=tmp;				
		}
	};
	//arithmic-plus
	template<int n,typename pointer,typename T>
	struct arithmic_plus {
		void operator()(pointer&& lhs,T rhs){ 
			std::get<n>(lhs)+=rhs;
			
			arithmic_plus<n-1,pointer,T> arith_r;
			arith_r(std::forward<pointer>(lhs),rhs);	
		}
	};
	template<typename pointer,typename T>
	struct arithmic_plus<0,pointer,T> {
		void operator()(pointer&& lhs,T rhs){
			std::get<0>(lhs)+=rhs;
		}
	};
	//arithmic-minus
	template<int n,typename pointer, typename T>
	struct arithmic_minus {
		void operator()(pointer&& lhs,const T& rhs){

			std::get<n>(lhs)-=rhs;
			
			arithmic_minus<n-1,pointer,T> arith_r;
			arith_r(std::forward<pointer>(lhs),rhs);	
		}
	};
	template<typename pointer,typename T>
	struct arithmic_minus<0,pointer,T> {
		void operator()(pointer&& lhs,const T& rhs){
			std::get<0>(lhs)-=rhs;
		}
	};
	
}















