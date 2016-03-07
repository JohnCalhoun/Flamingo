//dataframe.inl
#include "dataframe.cpp"
#include "functors.cpp"
template<class ... Type>
template<int n>
	typename column_return<n,Type...>::type* 
	dataframe<Type...>::column_access()
{
	typedef typename traits<Type...>::column_return<n>::type column; 
	
	columnbase*	base_ptr	=_column_array[n];
	column*	column_ptr	=static_cast<column*>(base_ptr);	
	return column_ptr;
};

//-------------------constructors/descrutors 
template<class ... Type>
	dataframe<Type...>::dataframe()
{
	dataframe_functors::construct_empty<traits<Type...>::_numCol-1,Type...> recurs;
	recurs(_column_array); 
};


template<class ... Type>
	dataframe<Type...>::dataframe(
		const dataframe<Type...>& other)
{	
	dataframe_functors::copy<traits<Type...>::_numCol-1,Type...> copier; 
	copier(_column_array,other._column_array); 
};



template<class ... Type>
	dataframe<Type...>::dataframe(
		dataframe<Type...>::size_type s,
		dataframe<Type...>::value_type v)
{
	dataframe_functors::fill<traits<Type...>::_numCol-1,Type...> filler;
	filler(_column_array,s,v); 
};

template<class ... Type>
	dataframe<Type...>::dataframe(
		dataframe<Type...>::size_type s)
{
	dataframe_functors::construct<traits<Type...>::_numCol-1,Type...> recurs;
	recurs(_column_array,s); 
};

template<class ... Type>
	dataframe<Type...>::dataframe(
		dataframe<Type...>::iterator start,
		dataframe<Type...>::iterator stop)
{

	dataframe_functors::it_copy<traits<Type...>::_numCol-1,Type...> it_copier;
	it_copier(
		_column_array,
		start,
		stop
		); 
};

template<class ... Type>
	dataframe<Type...>::~dataframe()
{
	dataframe_functors::destructor<traits<Type...>::_numCol-1,Type...> recurs;
	recurs(_column_array);
};

//-------------------container member functions-------------
//-------------------consts

template<class ... Type>
	dataframe<Type...>::reference 
	dataframe<Type...>::at(size_type n)const
{
	iterator start=begin(); 
	return *(start+n); 


};
template<class ... Type>
	dataframe<Type...>::reference	
	dataframe<Type...>::front()const
{
	return at(0); 
};
template<class ... Type>
	dataframe<Type...>::reference 
	dataframe<Type...>::back()const
{
	return at(size()-1);
};

template<class ... Type>
	dataframe<Type...>::iterator 
	dataframe<Type...>::begin()const
{
	iterator it(_column_array); 
	return it; 
};
/*
template<class ... Type>
	dataframe<Type...>::zip_it  
	dataframe<Type...>::begin_zip()const
{

};
*/
template<class ... Type>
	dataframe<Type...>::iterator 
	dataframe<Type...>::end()const
{
	iterator it=begin(); 
	it+=size()+1; 	
	return it;
}; 
/*
template<class ... Type>
	dataframe<Type...>::zip_it 
	dataframe<Type...>::end_zip()const
{

};
*/

template<class ... Type>
	dataframe<Type...>::size_type 
	dataframe<Type...>::size()const
{
	typedef typename traits<Type...>::Return<0>::type_base type; 
	size_type size;	

	column<type>* col_ptr=static_cast<column<type>*>(_column_array[0]);
	if(col_ptr){	
		size=col_ptr->size(); 
	}else{
		size=0;
	}
	return size;
};

template<class ... Type>
	dataframe<Type...>::size_type 
	dataframe<Type...>::max_size()const
{
	typedef typename traits<Type...>::Return<0>::type_base type; 
	size_type max_size;	

	column<type>* col_ptr=static_cast<column<type>*>(_column_array[0]);
	if(col_ptr){	
		max_size=col_ptr->max_size(); 
	}else{
		max_size=0;
	}
	return max_size;
};

template<class ... Type>
	bool 
	dataframe<Type...>::empty()const
{
	typedef typename traits<Type...>::Return<0>::type_base type; 
	bool empty;	

	column<type>* col_ptr=static_cast<column<type>*>(_column_array[0]);
	if(col_ptr){	
		empty=col_ptr->empty(); 
	}else{
		empty=0;
	}
	return empty;
};

template<class ... Type>
	void 
	dataframe<Type...>::reserve(dataframe<Type...>::size_type)
{

};

template<class ... Type>
	dataframe<Type...>::size_type 
	dataframe<Type...>::capacity()const
{
	typedef typename traits<Type...>::Return<0>::type_base type; 
	size_type cap;	

	column<type>* col_ptr=static_cast<column<type>*>(_column_array[0]);
	if(col_ptr){	
		cap=col_ptr->capacity(); 
	}else{
		cap=0;
	}
	return cap;
};


//-------------------non consts
template<class ... Type>
	void 
	dataframe<Type...>::assign(
		dataframe<Type...>::iterator start,
		dataframe<Type...>::iterator stop)
{
	clear(); 
	dataframe_functors::assign_range<traits<Type...>::_numCol-1,Type...> recurs;
	recurs(_column_array,start,stop); 
};
template<class ... Type>
	void 
	dataframe<Type...>::assign(
		dataframe<Type...>::size_type s,
		dataframe<Type...>::value_type v)
{
	clear(); 
	dataframe_functors::assign_value<traits<Type...>::_numCol-1,Type...> recurs;
	recurs(_column_array,s,v); 
};

template<class ... Type>
	void 
	dataframe<Type...>::clear()
{
	dataframe_functors::clear<traits<Type...>::_numCol-1,Type...> clearer;
	clearer(_column_array); 
};

template<class ... Type>
	dataframe<Type...>::iterator 
	dataframe<Type...>::insert(
		dataframe<Type...>::iterator pos,
		dataframe<Type...>::value_type v)
{
	size_type index=end()-pos; 
	if(index >= size() ){
		resize(index);
	}
	dataframe_functors::insert_value<traits<Type...>::_numCol-1,Type...> recurs;
	recurs(_column_array,pos,v); 
	return pos; 


};
template<class ... Type>
	dataframe<Type...>::iterator 
	dataframe<Type...>::insert(
		dataframe<Type...>::iterator pos,
		dataframe<Type...>::iterator start,
		dataframe<Type...>::iterator stop)
{
	dataframe_functors::insert_range<traits<Type...>::_numCol-1,Type...> recurs;
	recurs(_column_array,pos,start,stop); 
	return pos; 
};
template<class ... Type>
	dataframe<Type...>::iterator 
	dataframe<Type...>::erase(
		dataframe<Type...>::iterator pos)
{
	dataframe_functors::erase_value<traits<Type...>::_numCol-1,Type...> recurs;
	recurs(_column_array,pos);
	return pos;  
};
template<class ... Type>
	dataframe<Type...>::iterator 
	dataframe<Type...>::erase(
		dataframe<Type...>::iterator start,
		dataframe<Type...>::iterator stop)
{
	dataframe_functors::erase_range<traits<Type...>::_numCol-1,Type...> recurs;
	recurs(_column_array,start,stop); 
	return start; 
};
template<class ... Type>
	void 
	dataframe<Type...>::push_back(
		dataframe<Type...>::value_type value)
{
	insert(end()-1,value); 
};
template<class ... Type>
	void 
	dataframe<Type...>::pop_back()
{
	erase(end()-1); 
};
template<class ... Type>
	void 
	dataframe<Type...>::resize(
		dataframe<Type...>::size_type n)
{
	dataframe_functors::resize<traits<Type...>::_numCol-1,Type...> recurs;
	recurs(_column_array,n); 
};
template<class ... Type>
	void 
	dataframe<Type...>::resize(
		dataframe<Type...>::size_type n,
		dataframe<Type...>::value_type v)
{
	dataframe_functors::resize_value<traits<Type...>::_numCol-1,Type...> recurs;
	recurs(_column_array,n,v); 
};
/*	
template<class ... Type>
	void 
	dataframe<Type...>::swap(
		dataframe<Type...>&)
{

}; 
*/
//-------------------------operators
template<class ... Type>
	dataframe<Type...>::reference 
	dataframe<Type...>::operator[](
		dataframe<Type...>::size_type n)
{
	return at(n);
};
/*
template<class ... Type>
	dataframe<Type...>::reference 
	dataframe<Type...>::operator=(
		const dataframe<Type...>&)
{

};

template<class ... Type>
	bool 
	dataframe<Type...>::operator==(
		const dataframe<Type...>&)const
{

};
*/
/*
template<class ... Type>
	bool 
	dataframe<Type...>::operator!=(
		const dataframe<Type...>& other)const
{
	return !(*this==other); 	
};


*/











