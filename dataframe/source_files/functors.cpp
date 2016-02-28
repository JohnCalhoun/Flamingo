//functors.cpp
#include "traits.cpp"
#include "dataframe.cpp"
#include "columns.cpp"

namespace dataframe_functors{

	template<int n,class ... Type>
	struct copy {
		typedef typename dataframe<Type...>::branch branch;

		void operator()(branch& branch_1, const branch& branch_2){
			typedef typename traits<Type...>::Return<n>::type_base	type;
			typedef column<type>							Column; 

			Column* ptr_2=static_cast<Column*>(branch_2[n]); 

			if(ptr_2){
				Column* ptr_1=new Column; 
				*ptr_1=*ptr_2;
				branch_1[n]=static_cast<columnbase*>(ptr_1); 
			}else{
				Column* ptr_1=NULL; 
				branch_1[n]=static_cast<columnbase*>(ptr_1); 
			}
 		

			copy<n-1,Type...> copy_r;
			copy_r(branch_1,branch_2);	
		}
	};
	template<class ... Type>
	struct copy<0,Type...> {
		typedef typename dataframe<Type...>::branch branch;

		void operator()(branch& branch_1,const  branch& branch_2){
			typedef typename traits<Type...>::Return<0>::type_base	type;
			typedef column<type>			Column; 

			Column* ptr_2=static_cast<Column*>(branch_2[0]); 
	
			if(ptr_2){
				Column* ptr_1=new Column; 
				*ptr_1=*ptr_2;
				branch_1[0]=static_cast<columnbase*>(ptr_1); 
			}else{
				Column* ptr_1=NULL; 
				branch_1[0]=static_cast<columnbase*>(ptr_1); 
			}			
		}
	};
}
