//functors.cpp
#include "columns.cpp"

namespace dataframe_functors{

	template<int n,class ... Type>
	struct copy {
		typedef typename traits<Type...>::ColumnArray ColumnArray;

		void operator()(	ColumnArray&		column_array_1, 
						const ColumnArray&  column_array_2){
			typedef typename traits<Type...>::Return<n>::type_base	type;
			typedef column<type>							Column; 

			const int position=n; 

			Column* ptr_2=static_cast<Column*>(column_array_2[position]); 
			
			
			if(ptr_2){
				Column* ptr_1=new Column; 
				(*ptr_1)=(*ptr_2);
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
		typedef typename traits<Type...>::value_type value_type; 
		
		typedef typename traits<Type...>::Return<n>::type_base type; 
		typedef column<type> Column; 

		void operator()(	ColumnArray& column_array,
						size s,
						value_type v){
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

	template<int n,class ... Type>
	struct it_copy{
		typedef typename traits<Type...>::ColumnArray ColumnArray;
		typedef dataframe_iterator<Type...> iterator; 

		typedef typename traits<Type...>::size_type size_type;
		typedef typename traits<Type...>::value_type value_type; 

		typedef typename traits<Type...>::Return<0>::type_base type; 
		typedef column<type> Column; 


		void operator()(
			ColumnArray& columns,
			iterator start,
			iterator stop	
		){
			size_type size=stop-start;
			Column* ptr=new Column(size); 
	
			ptr->copy(
				start.template get<n>(),
				stop.template get<n>()); 	

			columns[n]=static_cast<columnbase*>(ptr); 

			it_copy<n-1,Type...> it_copy_r; 
			it_copy_r(columns,start,stop); 
		}
	};
	template<class ... Type>
	struct it_copy<0,Type...>{
		typedef typename traits<Type...>::ColumnArray ColumnArray;
		typedef typename dataframe<Type...>::iterator iterator; 

		typedef typename traits<Type...>::size_type size_type;
		typedef typename traits<Type...>::value_type value_type; 

		typedef typename traits<Type...>::Return<0>::type_base type; 
		typedef column<type> Column; 



		void operator()(
			ColumnArray& columns,
			iterator start,
			iterator stop	
		){
			size_type size=stop-start;
			Column* ptr=new Column(size); 
	
			ptr->copy(
				start.template get<0>(),
				stop.template get<0>()); 	

			columns[0]=static_cast<columnbase*>(ptr); 
		}
	};	
	template<int n,class ... Type>
	struct clear{
		typedef typename traits<Type...>::ColumnArray ColumnArray;
		typedef dataframe_iterator<Type...> iterator; 

		typedef typename traits<Type...>::size_type size_type;
		typedef typename traits<Type...>::value_type value_type; 

		typedef typename traits<Type...>::Return<0>::type_base type; 
		typedef column<type> Column; 

		void operator()(ColumnArray& columns){
			Column* ptr=static_cast<Column*>(columns[n]);
			if(ptr){
				ptr->clear(); 
			}

			clear<n-1,Type...> clear_r; 
			clear_r(columns); 
		}
	};
	template<class ... Type>
	struct clear<0,Type...>{
		typedef typename traits<Type...>::ColumnArray ColumnArray;
		typedef typename dataframe<Type...>::iterator iterator; 

		typedef typename traits<Type...>::size_type size_type;
		typedef typename traits<Type...>::value_type value_type; 

		typedef typename traits<Type...>::Return<0>::type_base type; 
		typedef column<type> Column; 

		void operator()(ColumnArray& columns){
			Column* ptr=static_cast<Column*>(columns[0]);
			if(ptr){
				ptr->clear(); 	
			}
		}
	};

	template<int n,class ... Type>
	struct assign_range{
		typedef typename traits<Type...>::ColumnArray ColumnArray;
		typedef typename dataframe<Type...>::iterator iterator; 

		typedef typename traits<Type...>::size_type size_type;
		typedef typename traits<Type...>::value_type value_type; 

		typedef typename traits<Type...>::Return<n>::type_base type; 
		typedef column<type> Column; 


		void operator()(ColumnArray& columnarray,iterator& start,iterator& stop){
			Column* ptr=static_cast<Column*>(columnarray[n]);
			ptr->assign(start,stop);

			assign_range<n-1,Type...> recursive;
			recursive(columnarray,start,stop); 
		}

	};
	template<class ... Type>
	struct assign_range<0,Type...>{
		typedef typename traits<Type...>::ColumnArray ColumnArray;
		typedef typename dataframe<Type...>::iterator iterator; 

		typedef typename traits<Type...>::size_type size_type;
		typedef typename traits<Type...>::value_type value_type; 

		typedef typename traits<Type...>::Return<0>::type_base type; 
		typedef column<type> Column; 


		void operator()(ColumnArray& columnarray,iterator& start,iterator& stop){
			Column* ptr=static_cast<Column*>(columnarray[0]);
			ptr->assign(start,stop);

		}

	};

	template<int n,class ... Type>
	struct assign_value{
		typedef typename traits<Type...>::ColumnArray ColumnArray;
		typedef typename dataframe<Type...>::iterator iterator; 

		typedef typename traits<Type...>::size_type size_type;
		typedef typename traits<Type...>::value_type value_type; 

		typedef typename traits<Type...>::Return<n>::type_base type; 
		typedef column<type> Column; 

		void operator()(ColumnArray& columnarray,size_type s,value_type v){
			Column* ptr=static_cast<Column*>(columnarray[n]);

			ptr->assign(s,v); 

			assign_value<n-1,Type...> recursive;
			recursive(columnarray,s,v); 
		}

	};
	template<class ... Type>
	struct assign_value<0,Type...>{
		typedef typename traits<Type...>::ColumnArray ColumnArray;
		typedef typename dataframe<Type...>::iterator iterator; 

		typedef typename traits<Type...>::size_type size_type;
		typedef typename traits<Type...>::value_type value_type; 

		typedef typename traits<Type...>::Return<0>::type_base type; 
		typedef column<type> Column; 

		void operator()(ColumnArray& columnarray,size_type s,value_type v){
			Column* ptr=static_cast<Column*>(columnarray[0]);

			ptr->assign(s,v); 
		}


	};

	template<int n,class ... Type>
	struct insert_range{
		typedef typename traits<Type...>::ColumnArray ColumnArray;
		typedef typename dataframe<Type...>::iterator iterator; 

		typedef typename traits<Type...>::size_type size_type;
		typedef typename traits<Type...>::value_type value_type; 

		typedef typename traits<Type...>::Return<n>::type_base type; 
		typedef column<type> Column; 

		void operator()(	ColumnArray& columnarray,
						iterator& pos,
						iterator& start, 
						iterator& stop){
			Column* ptr=static_cast<Column*>(columnarray[n]);
			
			ptr->insert(pos,start,stop);

			insert_range<n-1,Type...> recursive;
			recursive(columnarray,pos,start); 
		}

	};
	template<class ... Type>
	struct insert_range<0,Type...>{
		typedef typename traits<Type...>::ColumnArray ColumnArray;
		typedef typename dataframe<Type...>::iterator iterator; 

		typedef typename traits<Type...>::size_type size_type;
		typedef typename traits<Type...>::value_type value_type; 

		typedef typename traits<Type...>::Return<0>::type_base type; 
		typedef column<type> Column; 

		void operator()(
				ColumnArray& columnarray,
				iterator& pos,
				iterator& start, 
				iterator& stop)
		{
			Column* ptr=static_cast<Column*>(columnarray[0]);

			ptr->insert(pos,start,stop);


		}
	};

	template<int n,class ... Type>
	struct insert_value{
		typedef typename traits<Type...>::ColumnArray ColumnArray;
		typedef typename dataframe<Type...>::iterator iterator; 

		typedef typename traits<Type...>::size_type size_type;
		typedef typename traits<Type...>::value_type value_type; 

		typedef typename traits<Type...>::Return<n>::type_base type; 
		typedef column<type> Column; 

		void operator()(ColumnArray& columnarray,iterator& pos, value_type& v){
			Column* ptr=static_cast<Column*>(columnarray[n]);

			ptr->insert(
						pos.template get<n>() ,
						std::get<n>(v)); 			

			insert_value<n-1,Type...> recursive;
			recursive(columnarray,pos,v); 

		}

	};
	template<class ... Type>
	struct insert_value<0,Type...>{
		typedef typename traits<Type...>::ColumnArray ColumnArray;
		typedef typename dataframe<Type...>::iterator iterator; 

		typedef typename traits<Type...>::size_type size_type;
		typedef typename traits<Type...>::value_type value_type; 

		typedef typename traits<Type...>::Return<0>::type_base type; 
		typedef column<type> Column; 

		void operator()(ColumnArray& columnarray,iterator& pos, value_type& v){
			Column* ptr=static_cast<Column*>(columnarray[0]);

			ptr->insert(
					pos.template get<0>() ,
					std::get<0>(v)
					); 			
		}


	};

	template<int n,class ... Type>
	struct erase_range{
		typedef typename traits<Type...>::ColumnArray ColumnArray;
		typedef typename dataframe<Type...>::iterator iterator; 

		typedef typename traits<Type...>::size_type size_type;
		typedef typename traits<Type...>::value_type value_type; 

		typedef typename traits<Type...>::Return<n>::type_base type; 
		typedef column<type> Column; 

		void operator()(ColumnArray& columnarray,iterator& start, iterator& stop){
			Column* ptr=static_cast<Column*>(columnarray[n]);	
			ptr->erase(start,stop); 
	
			erase_range<n-1,Type...> recursive;
			recursive(columnarray,start,stop); 

		}

	};
	template<class ... Type>
	struct erase_range<0,Type...>{
		typedef typename traits<Type...>::ColumnArray ColumnArray;
		typedef typename dataframe<Type...>::iterator iterator; 

		typedef typename traits<Type...>::size_type size_type;
		typedef typename traits<Type...>::value_type value_type; 

		typedef typename traits<Type...>::Return<0>::type_base type; 
		typedef column<type> Column; 

		void operator()(ColumnArray& columnarray,iterator& start, iterator& stop){
			Column* ptr=static_cast<Column*>(columnarray[0]);

			ptr->erase(start,stop); 
	
		}


	};

	template<int n,class ... Type>
	struct erase_value{
		typedef typename traits<Type...>::ColumnArray ColumnArray;
		typedef typename dataframe<Type...>::iterator iterator; 

		typedef typename traits<Type...>::size_type size_type;
		typedef typename traits<Type...>::value_type value_type; 

		typedef typename traits<Type...>::Return<n>::type_base type; 
		typedef column<type> Column; 

		void operator()(ColumnArray& columnarray,iterator& pos){
			Column* ptr=static_cast<Column*>(columnarray[n]);
		
			ptr->erase(pos);

			erase_value<n-1,Type...> recursive;
			recursive(columnarray,pos); 

		}

	};
	template<class ... Type>
	struct erase_value<0,Type...>{
		typedef typename traits<Type...>::ColumnArray ColumnArray;
		typedef typename dataframe<Type...>::iterator iterator; 

		typedef typename traits<Type...>::size_type size_type;
		typedef typename traits<Type...>::value_type value_type; 

		typedef typename traits<Type...>::Return<0>::type_base type; 
		typedef column<type> Column; 

		void operator()(ColumnArray& columnarray,iterator& pos){
			Column* ptr=static_cast<Column*>(columnarray[0]);

			ptr->erase(pos);
		}
	};

	template<int n,class ... Type>
	struct resize{
		typedef typename traits<Type...>::ColumnArray ColumnArray;
		typedef typename dataframe<Type...>::iterator iterator; 

		typedef typename traits<Type...>::size_type size_type;
		typedef typename traits<Type...>::value_type value_type; 

		typedef typename traits<Type...>::Return<n>::type_base type; 
		typedef column<type> Column; 

		void operator()(ColumnArray& columnarray,size_type x){
			Column* ptr=static_cast<Column*>(columnarray[n]);
		
			ptr->resize(x); 

			resize<n-1,Type...> recursive;
			recursive(columnarray,x); 

		}

	};
	template<class ... Type>
	struct resize<0,Type...>{
		typedef typename traits<Type...>::ColumnArray ColumnArray;
		typedef typename dataframe<Type...>::iterator iterator; 

		typedef typename traits<Type...>::size_type size_type;
		typedef typename traits<Type...>::value_type value_type; 

		typedef typename traits<Type...>::Return<0>::type_base type; 
		typedef column<type> Column; 

		void operator()(ColumnArray& columnarray,size_type n){
			Column* ptr=static_cast<Column*>(columnarray[0]);

			ptr->resize(n); 

		}


	};

	template<int n,class ... Type>
	struct resize_value{
		typedef typename traits<Type...>::ColumnArray ColumnArray;
		typedef typename dataframe<Type...>::iterator iterator; 

		typedef typename traits<Type...>::size_type size_type;
		typedef typename traits<Type...>::value_type value_type; 

		typedef typename traits<Type...>::Return<n>::type_base type; 
		typedef column<type> Column; 

		void operator()(ColumnArray& columnarray,size_type x,value_type v){
			Column* ptr=static_cast<Column*>(columnarray[n]);
			
			ptr->resize(x,v);

			resize_value<n-1,Type...> recursive;
			recursive(columnarray,x,v); 

		}

	};
	template<class ... Type>
	struct resize_value<0,Type...>{
		typedef typename traits<Type...>::ColumnArray ColumnArray;
		typedef typename dataframe<Type...>::iterator iterator; 

		typedef typename traits<Type...>::size_type size_type;
		typedef typename traits<Type...>::value_type value_type; 

		typedef typename traits<Type...>::Return<0>::type_base type; 
		typedef column<type> Column; 

		void operator()(ColumnArray& columnarray,size_type n,value_type v){
			Column* ptr=static_cast<Column*>(columnarray[0]);

			ptr->resize(n,v);
		}
	};

	template<int n,class ... Type>
	struct construct {
		typedef typename traits<Type...>::ColumnArray ColumnArray;
		typedef typename traits<Type...>::size_type size_type; 

		void operator()(	ColumnArray&		column_array, 
						size_type size){
			typedef typename traits<Type...>::Return<n>::type_base	type;
			typedef column<type>							Column; 

			const int position=n; 
				

			Column* ptr=new Column(size); 		
			column_array[position]=static_cast<columnbase*>(ptr); 

			construct<n-1,Type...> recurs;
			recurs(column_array,size);	
		}
	};
	template<class ... Type>
	struct construct<0,Type...> {
		typedef typename traits<Type...>::ColumnArray ColumnArray;
		typedef typename traits<Type...>::size_type size_type; 

		void operator()(	ColumnArray&		column_array,
						size_type size){
			typedef typename traits<Type...>::Return<0>::type_base	type;
			typedef column<type>			Column; 
	
			const int position=0; 			

			Column* ptr=new Column(size); 		
			column_array[position]=static_cast<columnbase*>(ptr); 
		}
	};
	template<int n,class ... Type>
	struct construct_empty {
		typedef typename traits<Type...>::ColumnArray ColumnArray;
		typedef typename traits<Type...>::size_type size_type; 

		void operator()(	ColumnArray&		column_array){
			typedef typename traits<Type...>::Return<n>::type_base	type;
			typedef column<type>							Column; 

			const int position=n; 
				

			Column* ptr=new Column(); 		
			column_array[position]=static_cast<columnbase*>(ptr); 

			construct_empty<n-1,Type...> recurs;
			recurs(column_array);	
		}
	};
	template<class ... Type>
	struct construct_empty<0,Type...> {
		typedef typename traits<Type...>::ColumnArray ColumnArray;
		typedef typename traits<Type...>::size_type size_type; 

		void operator()(	ColumnArray&		column_array){
			typedef typename traits<Type...>::Return<0>::type_base	type;
			typedef column<type>			Column; 
	
			const int position=0; 			

			Column* ptr=new Column(); 		
			column_array[position]=static_cast<columnbase*>(ptr); 
		}
	};
	template<int n,class ... Type>
	struct destructor {
		typedef typename traits<Type...>::ColumnArray ColumnArray;
		typedef typename traits<Type...>::size_type size_type; 

		void operator()(	ColumnArray&		column_array){
			typedef typename traits<Type...>::Return<n>::type_base	type;
			typedef column<type>							Column; 


			Column* ptr=static_cast<Column*>(column_array[n]); 
			if(ptr){
				delete ptr; 
			}

			destructor<n-1,Type...> recurs;
			recurs(column_array);	
		}
	};
	template<class ... Type>
	struct destructor<0,Type...> {
		typedef typename traits<Type...>::ColumnArray ColumnArray;
		typedef typename traits<Type...>::size_type size_type; 

		void operator()(	ColumnArray&		column_array){
			typedef typename traits<Type...>::Return<0>::type_base	type;
			typedef column<type>			Column; 
	
			Column* ptr=static_cast<Column*>(column_array[0]); 
			if(ptr){
				delete ptr; 
			}

		}
	};
}
