//functors.cpp
#include "columns.cpp"

namespace dataframe_functors{

	template<int n,class ... Type>
	struct fill {
		typedef typename dataframe<Type...>::ColumnTuple ColumnTuple;

		typedef typename traits<Type...>::size_type size_type;
		typedef typename traits<Type...>::value_type value_type; 
		
		typedef typename column_tuple<Type...>::element<n>::type Column; 

		void operator()(	ColumnTuple&& column_tuple,
						size_type s,
						value_type v){

			Column& column=std::get<n>(column_tuple);
			column.resize(s);
			column.fill(std::get<n>(v));  
		
			fill<n-1,Type...> recursive;
			recursive(std::forward<ColumnTuple>(column_tuple),s,v); 
		}
	};
	template<class ... Type>
	struct fill<0,Type...> {
			typedef typename dataframe<Type...>::ColumnTuple ColumnTuple;

		typedef typename traits<Type...>::size_type size_type;
		typedef typename traits<Type...>::value_type value_type; 
		
		typedef typename column_tuple<Type...>::element<0>::type Column; 

		void operator()(	ColumnTuple&& column_tuple,
						size_type s,
						value_type v){

			Column& column=std::get<0>(column_tuple);
			column.resize(s);
			column.fill(std::get<0>(v));  
		}
	};

	template<int n,class ... Type>
	struct it_copy{
		typedef typename dataframe<Type...>::ColumnTuple ColumnTuple;
		typedef typename traits<Type...>::size_type size;	
		typedef typename column_tuple<Type...>::element<n>::type Column; 
		typedef typename dataframe<Type...>::iterator iterator; 
		typedef typename traits<Type...>::value_type	value_type; 

		void operator()(
			ColumnTuple&& column_tuple,
			iterator start,
			iterator stop	
		){
			Column& column=std::get<n>(column_tuple);

			column.copy(
				start.template get<n>(),
				stop.template get<n>()); 	

			it_copy<n-1,Type...> recusive; 
			recusive(std::forward<ColumnTuple>(column_tuple),start,stop); 
		}
	};
	template<class ... Type>
	struct it_copy<0,Type...>{
		typedef typename dataframe<Type...>::ColumnTuple ColumnTuple;
		typedef typename traits<Type...>::size_type size;	
		typedef typename column_tuple<Type...>::element<0>::type Column; 
		typedef typename dataframe<Type...>::iterator iterator; 
		typedef typename traits<Type...>::value_type	value_type; 

		void operator()(
			ColumnTuple&& column_tuple,
			iterator start,
			iterator stop	
		){
			Column& column=std::get<0>(column_tuple);

			column.copy(
				start.template get<0>(),
				stop.template get<0>()); 	
		}
	};	

	template<int n,class ... Type>
	struct clear{
		typedef typename dataframe<Type...>::ColumnTuple ColumnTuple;
		typedef typename column_tuple<Type...>::element<n>::type Column;

		void operator()(ColumnTuple&& columnTuple){
			Column& column=std::get<n>(columnTuple); 
			column.clear(); 

			clear<n-1,Type...> clear_r; 
			clear_r(std::forward<ColumnTuple>(columnTuple)); 
		}
	};
	template<class ... Type>
	struct clear<0,Type...>{
		typedef typename dataframe<Type...>::ColumnTuple ColumnTuple;
		typedef typename column_tuple<Type...>::element<0>::type Column;

		void operator()(ColumnTuple&& columnTuple){
			Column& column=std::get<0>(columnTuple); 
			column.clear(); 
		}
	};

	template<int n,class ... Type>
	struct assign_range{
		typedef typename dataframe<Type...>::ColumnTuple ColumnTuple;
		typedef typename dataframe<Type...>::iterator iterator; 
		typedef typename column_tuple<Type...>::element<n>::type Column;

		void operator()(
				ColumnTuple&& columnTuple,
				iterator& start,
				iterator& stop)
		{
			Column& column=std::get<n>(columnTuple); 
			column.assign(	start.template get<n>(),
						stop.template get<n>());

			assign_range<n-1,Type...> recursive;
			recursive(	std::forward<ColumnTuple>(columnTuple),
						start,
						stop); 
		}

	};
	template<class ... Type>
	struct assign_range<0,Type...>{
		typedef typename dataframe<Type...>::ColumnTuple ColumnTuple;
		typedef typename dataframe<Type...>::iterator iterator; 
		typedef typename column_tuple<Type...>::element<0>::type Column;

		void operator()(
				ColumnTuple&& columnTuple,
				iterator& start,
				iterator& stop)
		{
			Column& column=std::get<0>(columnTuple); 
			column.assign(	start.template get<0>(),
						stop.template get<0>());
		}

	};
	template<int n,class ... Type>
	struct assign_value{
		typedef typename dataframe<Type...>::ColumnTuple ColumnTuple;
		typedef typename column_tuple<Type...>::element<n>::type Column;
		typedef typename traits<Type...>::size_type size_type;
		typedef typename traits<Type...>::value_type value_type; 

		void operator()(	ColumnTuple&& columnTuple,
						size_type s,
						value_type v)
		{
			Column& column=std::get<n>(columnTuple); 
			column.assign(s,std::get<n>(v)); 

			assign_value<n-1,Type...> recursive;
			recursive(std::forward<ColumnTuple>(columnTuple),s,v); 
		}
	};
	template<class ... Type>
	struct assign_value<0,Type...>{
		typedef typename dataframe<Type...>::ColumnTuple ColumnTuple;
		typedef typename column_tuple<Type...>::element<0>::type Column;
		typedef typename traits<Type...>::size_type size_type;
		typedef typename traits<Type...>::value_type value_type; 

		void operator()(	ColumnTuple&& columnTuple,
						size_type s,
						value_type v)
		{
			Column& column=std::get<0>(columnTuple); 
			column.assign(s,std::get<0>(v)); 
		}
	};
	template<int n,class ... Type>
	struct insert_range{
		typedef typename dataframe<Type...>::ColumnTuple ColumnTuple;
		typedef typename dataframe<Type...>::iterator iterator; 
		typedef typename column_tuple<Type...>::element<n>::type Column;

		void operator()(	ColumnTuple&& columnTuple,
						iterator& pos,
						iterator& start, 
						iterator& stop){
			Column& column=std::get<n>(columnTuple); 	
			column.insert(	pos.template get<n>(),
						start.template get<n>(),
						stop.template get<n>()
			);

			insert_range<n-1,Type...> recursive;
			recursive(	std::forward<ColumnTuple>(columnTuple),
						pos,
						start,
						stop); 
		}

	};
	template<class ... Type>
	struct insert_range<0,Type...>{
		typedef typename dataframe<Type...>::ColumnTuple ColumnTuple;
		typedef typename dataframe<Type...>::iterator iterator; 
		typedef typename column_tuple<Type...>::element<0>::type Column;

		void operator()(	ColumnTuple&& columnTuple,
						iterator& pos,
						iterator& start, 
						iterator& stop){
			Column& column=std::get<0>(columnTuple); 	
			column.insert(	pos.template get<0>(),
						start.template get<0>(),
						stop.template get<0>()
			);
		}
	};


	template<int n,class ... Type>
	struct insert_value{
		typedef typename dataframe<Type...>::ColumnTuple ColumnTuple;
		typedef typename dataframe<Type...>::iterator iterator; 
		typedef typename column_tuple<Type...>::element<n>::type Column;
		typedef typename traits<Type...>::value_type value_type; 

		void operator()(	ColumnTuple&& columnTuple,
						iterator& pos, 
						value_type& v)
		{
			Column& column=std::get<n>(columnTuple); 	
			column.insert(
						pos.template get<n>(),
						std::get<n>(v)
			); 			

			insert_value<n-1,Type...> recursive;
			recursive(std::forward<ColumnTuple>(columnTuple),pos,v); 
		}

	};

	template<class ... Type>
	struct insert_value<0,Type...>{
		typedef typename dataframe<Type...>::ColumnTuple ColumnTuple;
		typedef typename dataframe<Type...>::iterator iterator; 
		typedef typename column_tuple<Type...>::element<0>::type Column;
		typedef typename traits<Type...>::value_type value_type; 

		void operator()(	ColumnTuple&& columnTuple,
						iterator& pos, 
						value_type& v)
		{
			Column& column=std::get<0>(columnTuple); 	
			column.insert(
						pos.template get<0>() ,
						std::get<0>(v)); 			
		}

	};

	template<int n,class ... Type>
	struct erase_range{
		typedef typename dataframe<Type...>::ColumnTuple ColumnTuple;
		typedef typename dataframe<Type...>::iterator iterator; 
		typedef typename column_tuple<Type...>::element<n>::type Column;

		void operator()(	ColumnTuple&& columnTuple,
						iterator& start, 
						iterator& stop)	
		{
			Column& column=std::get<n>(columnTuple); 	
			column.erase(	start.template get<n>(),
						stop.template get<n>()); 
	
			erase_range<n-1,Type...> recursive;
			recursive(std::forward<ColumnTuple>(columnTuple),start,stop); 
		}
	};
	template<class ... Type>
	struct erase_range<0,Type...>{
		typedef typename dataframe<Type...>::ColumnTuple ColumnTuple;
		typedef typename dataframe<Type...>::iterator iterator; 
		typedef typename column_tuple<Type...>::element<0>::type Column;

		void operator()(	ColumnTuple&& columnTuple,
						iterator& start, 
						iterator& stop)	
		{
			Column& column=std::get<0>(columnTuple); 	
			column.erase(	start.template get<0>(),
						stop.template get<0>()); 
		}
	};

	template<int n,class ... Type>
	struct erase_value{
		typedef typename dataframe<Type...>::ColumnTuple ColumnTuple;
		typedef typename dataframe<Type...>::iterator iterator; 
		typedef typename column_tuple<Type...>::element<n>::type Column;

		void operator()(	ColumnTuple&& columnTuple,
						iterator& pos)
		{
			Column& column=std::get<n>(columnTuple); 	
			column.erase(pos.template get<n>());

			erase_value<n-1,Type...> recursive;
			recursive(std::forward<ColumnTuple>(columnTuple),pos); 
		}
	};
	template<class ... Type>
	struct erase_value<0,Type...>{
		typedef typename dataframe<Type...>::ColumnTuple ColumnTuple;
		typedef typename dataframe<Type...>::iterator iterator; 
		typedef typename column_tuple<Type...>::element<0>::type Column;

		void operator()(	ColumnTuple&& columnTuple,
						iterator& pos)
		{
			Column& column=std::get<0>(columnTuple); 		
			column.erase(pos.template get<0>());
		}

	};
	template<int n,class ... Type>
	struct resize{
		typedef typename dataframe<Type...>::ColumnTuple ColumnTuple;
		typedef typename column_tuple<Type...>::element<n>::type Column;
		typedef typename traits<Type...>::size_type size_type;

		void operator()(	ColumnTuple&& columnTuple,
						size_type x)
		{
			Column& column=std::get<n>(columnTuple); 		
			column.resize(x); 

			resize<n-1,Type...> recursive;
			recursive(std::forward<ColumnTuple>(columnTuple),x); 
		}
	};
	template<class ... Type>
	struct resize<0,Type...>{
		typedef typename dataframe<Type...>::ColumnTuple ColumnTuple;
		typedef typename column_tuple<Type...>::element<0>::type Column;
		typedef typename traits<Type...>::size_type size_type;

		void operator()(	ColumnTuple&& columnTuple,
						size_type x)
		{
			Column& column=std::get<0>(columnTuple); 		
			column.resize(x); 
		}
	};

	template<int n,class ... Type>
	struct resize_value{
		typedef typename dataframe<Type...>::ColumnTuple ColumnTuple;
		typedef typename column_tuple<Type...>::element<n>::type Column;
		typedef typename traits<Type...>::size_type size_type;
		typedef typename traits<Type...>::value_type value_type; 

		void operator()(	ColumnTuple&& columnTuple,
						size_type x,
						value_type v)
		{
			Column& column=std::get<n>(columnTuple); 		
			column.resize(x,v);

			resize_value<n-1,Type...> recursive;
			recursive(std::forward<ColumnTuple>(columnTuple),x,v); 
		}
	};
	template<class ... Type>
	struct resize_value<0,Type...>{
		typedef typename dataframe<Type...>::ColumnTuple ColumnTuple;
		typedef typename column_tuple<Type...>::element<0>::type Column;
		typedef typename traits<Type...>::size_type size_type;
		typedef typename traits<Type...>::value_type value_type; 

		void operator()(	ColumnTuple&& columnTuple,
						size_type x,
						value_type v)
		{
			Column& column=std::get<0>(columnTuple); 		
			column.resize(x,v);
		}
	};

	template<int n,class ... Type>
	struct construct {
		typedef typename dataframe<Type...>::ColumnTuple ColumnTuple;
		typedef typename traits<Type...>::size_type size;	
		typedef typename column_tuple<Type...>::element<n>::type Column; 

		void operator()(	ColumnTuple&& column_tuple,
						size s){

			Column& column=std::get<n>(column_tuple);
			column.resize(s);

			construct<n-1,Type...> recursive;
			recursive(std::forward<ColumnTuple>(column_tuple),s); 
		}

	};
	template<class ... Type>
	struct construct<0,Type...> {
		typedef typename dataframe<Type...>::ColumnTuple ColumnTuple;
		typedef typename traits<Type...>::size_type size;	
		typedef typename column_tuple<Type...>::element<0>::type Column; 

		void operator()(	ColumnTuple&& column_tuple,
						size s){

			Column& column=std::get<0>(column_tuple);
			column.resize(s);
		}
	};
	template<int n,class ... Type>
	struct reserve {
		typedef typename dataframe<Type...>::ColumnTuple ColumnTuple;
		typedef typename traits<Type...>::size_type size_type;	
		typedef typename column_tuple<Type...>::element<n>::type Column; 

		void operator()(	ColumnTuple&& column_tuple,
						size_type s)
		{
			Column& column=std::get<n>(column_tuple);
			column.reserve(s);

			construct<n-1,Type...> recursive;
			recursive(std::forward<ColumnTuple>(column_tuple),s); 
		}
	};
	template<class ... Type>
	struct reserve<0,Type...> {
		typedef typename dataframe<Type...>::ColumnTuple ColumnTuple;
		typedef typename traits<Type...>::size_type size_type;	
		typedef typename column_tuple<Type...>::element<0>::type Column; 

		void operator()(	ColumnTuple&& column_tuple,
						size_type s)
		{
			Column& column=std::get<0>(column_tuple);
			column.reserve(s);
		}
	};
}
