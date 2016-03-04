//dataframe.inl
#include "dataframe.cpp"
#include "functors.cpp"
/*
//-----------------private member functions
template<class ... Type>
	dataframe<Type...>::iterator 
	dataframe<Type...>::row_access(dataframe<Type...>::size_type n){

}; 		
*/
template<class ... Type>
template<int n>
	typename traits<Type...>::column_return<n>::type* 
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
	_column_array.fill(NULL);	
};


template<class ... Type>
	dataframe<Type...>::dataframe(
		const dataframe<Type...>& other)
{	
	dataframe_functors::copy<traits<Type...>::_numCol,Type...> copier; 
	copier(_column_array,other._column_array); 
};



template<class ... Type>
	dataframe<Type...>::dataframe(
		dataframe<Type...>::size_type,
		dataframe<Type...>::value_type)
{

};

template<class ... Type>
	dataframe<Type...>::dataframe(
		dataframe<Type...>::iterator,
		dataframe<Type...>::iterator)
{

};

template<class ... Type>
	dataframe<Type...>::~dataframe()
{
	
};

/* 
//-------------------container member functions-------------
//-------------------consts
template<class ... Type>
	dataframe<Type...>::reference 
	dataframe<Type...>::at()const
{

};
template<class ... Type>
	dataframe<Type...>::reference	
	dataframe<Type...>::front()const
{

};
template<class ... Type>
	dataframe<Type...>::reference 
	dataframe<Type...>::back()const
{

};
*/
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
/*
template<class ... Type>
	dataframe<Type...>::size_type 
	dataframe<Type...>::max_size()const
{
	

};
template<class ... Type>
	bool 
	dataframe<Type...>::empty()const
{

};
template<class ... Type>
	void 
	dataframe<Type...>::reserve(dataframe<Type...>::size_type)const
{

};
template<class ... Type>
	dataframe<Type...>::size_type 
	dataframe<Type...>::capacity()const
{

};
*/
/*
//-------------------non consts
template<class ... Type>
	void 
	dataframe<Type...>::assign(
		dataframe<Type...>::iterator,
		dataframe<Type...>::iterator)
{

};
template<class ... Type>
	void 
	dataframe<Type...>::assign(
		dataframe<Type...>::size_type,
		dataframe<Type...>::value_type)
{

};
template<class ... Type>
	void 
	dataframe<Type...>::clear()
{

};
template<class ... Type>
	dataframe<Type...>::iterator 
	dataframe<Type...>::insert(
		dataframe<Type...>::iterator,
		dataframe<Type...>::value_type)
{

};
template<class ... Type>
	dataframe<Type...>::iterator 
	dataframe<Type...>::insert(
		dataframe<Type...>::iterator,
		dataframe<Type...>::iterator,
		dataframe<Type...>::iterator)
{

};
template<class ... Type>
	dataframe<Type...>::iterator 
	dataframe<Type...>::erase(
		dataframe<Type...>::iterator)
{

};
template<class ... Type>
	dataframe<Type...>::iterator 
	dataframe<Type...>::erase(
		dataframe<Type...>::iterator,
		dataframe<Type...>::iterator)
{

};
template<class ... Type>
	void 
	dataframe<Type...>::push_back(
		dataframe<Type...>::value_type)
{

};
template<class ... Type>
	void 
	dataframe<Type...>::push_front(
		dataframe<Type...>::value_type)
{

};
template<class ... Type>
	void 
	dataframe<Type...>::pop_back()
{

};
template<class ... Type>
	void 
	dataframe<Type...>::pop_front()
{

};
template<class ... Type>
	void 
	dataframe<Type...>::resize(
		dataframe<Type...>::size_type)
{

};
template<class ... Type>
	void 
	dataframe<Type...>::resize(
		dataframe<Type...>::size_type,
		dataframe<Type...>::value_type)
{

};	
template<class ... Type>
	void 
	dataframe<Type...>::swap(
		dataframe<Type...>&)
{

}; 

//-------------------------operators
template<class ... Type>
	dataframe<Type...>::reference 
	dataframe<Type...>::operator[](
		dataframe<Type...>::size_type)
{

};
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
template<class ... Type>
	bool 
	dataframe<Type...>::operator!=(
		const dataframe<Type...>&)const
{

};


*/











