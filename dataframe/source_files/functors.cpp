//functors.cpp
#include "traits.cpp"
#include "dataframe.cpp"
#include "columns.cpp"

namespace dataframe_functors{

	template<int n,class ... Type>
	struct copy {
		typedef typename dataframe<Type...>::ColumnArray ColumnArray;

		void operator()(	ColumnArray&		column_array_1, 
						const ColumnArray& column_array_2){
			typedef typename traits<Type...>::Return<n>::type_base	type;
			typedef column<type>							Column; 

			Column* ptr_2=static_cast<Column*>(column_array_2[n]); 
			
			
			if(ptr_2){
				Column* ptr_1=new Column; 
				*ptr_1=*ptr_2;
				column_array_1[n]=static_cast<columnbase*>(ptr_1); 
			}else{
				Column* ptr_1=NULL; 
				column_array_1[n]=static_cast<columnbase*>(ptr_1); 
			}
 		

			copy<n-1,Type...> copy_r;
			copy_r(column_array_1,column_array_2);	
		}
	};
	template<class ... Type>
	struct copy<0,Type...> {
		typedef typename dataframe<Type...>::ColumnArray ColumnArray;

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
}
