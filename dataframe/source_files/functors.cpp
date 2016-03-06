//functors.cpp
#include "columns.cpp"

namespace dataframe_functors{

	template<int n,class ... Type>
	struct copy {
		typedef typename traits<Type...>::ColumnArray ColumnArray;

		void operator()(	ColumnArray&		column_array_1, 
						const ColumnArray& column_array_2){
			typedef typename traits<Type...>::Return<n>::type_base	type;
			typedef column<type>							Column; 

			const int position=n; 

			Column* ptr_2=static_cast<Column*>(column_array_2[position]); 
			
			
			if(ptr_2){
				Column* ptr_1=new Column; 
				*ptr_1=*ptr_2;
				column_array_1[position]=static_cast<columnbase*>(ptr_1); 
			}else{
				Column* ptr_1=NULL; 
				column_array_1[position]=static_cast<columnbase*>(ptr_1); 
			}
 		

			copy<n-1,Type...> copy_r;
			copy_r(column_array_1,column_array_2);	
		}
	};
	template<class ... Type>
	struct copy<0,Type...> {
		typedef typename traits<Type...>::ColumnArray ColumnArray;

		void operator()(	ColumnArray&		column_array_1,
						const ColumnArray&	column_array_2){
			typedef typename traits<Type...>::Return<0>::type_base	type;
			typedef column<type>			Column; 

			Column* ptr_2=static_cast<Column*>(column_array_2[0]); 
	
			if(ptr_2){
				Column* ptr_1=new Column; 
				*ptr_1=*ptr_2;
				column_array_1[0]=static_cast<columnbase*>(ptr_1); 
			}else{
				Column* ptr_1=NULL; 
				column_array_1[0]=static_cast<columnbase*>(ptr_1); 
			}			
		}
	};
	template<int n,class ... Type>
	struct fill {
		typedef typename traits<Type...>::ColumnArray ColumnArray;

		typedef typename traits<Type...>::size_type size;
		typedef typename traits<Type...>::value_type value; 
		
		typedef typename traits<Type...>::Return<n>::type_base type; 
		typedef column<type> Column; 

		void operator()(	ColumnArray& column_array,
						size s,
						value v){
			Column* ptr=new Column(s); 
			ptr->fill(std::get<n>(v));
			column_array[n]=static_cast<columnbase*>(ptr);		
		
			fill<n-1,Type...> fill_r;
			fill_r(column_array,s,v); 
		}
	};
	template<class ... Type>
	struct fill<0,Type...> {
		typedef typename traits<Type...>::ColumnArray ColumnArray;

		typedef typename traits<Type...>::size_type size;
		typedef typename traits<Type...>::value_type value; 

		typedef typename traits<Type...>::Return<0>::type_base type; 
		typedef column<type> Column; 


		void operator()(	ColumnArray& column_array,
						size s,
						value v){
			Column* ptr=new Column(s); 
			ptr->fill(std::get<0>(v));
			column_array[0]=static_cast<columnbase*>(ptr);		
	
		}
	};



















}
