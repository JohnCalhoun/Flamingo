//dataframe.inl
#include "dataframe.cpp"
#include "dataframe_functors.cpp"

//--------------------base------------
dataframeBase::dataframeBase(){
	_key=_addressbook.insert(this); 
}
dataframeBase::~dataframeBase(){
	_addressbook.remove(id()); 
}
dataframeBase::Value dataframeBase::find(dataframeBase::Key key){
	return _addressbook.find(key); 
}
dataframeBase::Key dataframeBase::id(){
	return _key; 
}
void dataframeBase::id(int x){
	_addressbook.change(id(),x); 
	_key=x; 
}

//-----------------dataframe
template<class ... Type>
template<int n>
	typename column_tuple<Type...>::element<n>::type& 
	dataframe<Type...>::column_access()
{
	typedef typename column_tuple<Type...>::element<n>::type		column; 
	
	column& col=std::get<n>(_column_tuple);
	return col;
};

template<class ... Type>
dataframe<Type...>::ColumnTuple&	dataframe<Type...>::tuple()
{	
	return _column_tuple;
};

template<class ... Type>
const dataframe<Type...>::ColumnTuple&	dataframe<Type...>::tuple_const()const
{	
	return _column_tuple;
};

//-------------------constructors/descrutors 
template<class ... Type>
	dataframe<Type...>::dataframe(){};

template<class ... Type>
	dataframe<Type...>::dataframe(
		const dataframe<Type...>& other)
{	
	tuple()=other.tuple_const(); 
};


template<class ... Type>
	dataframe<Type...>::dataframe(
		dataframe<Type...>::size_type s,
		dataframe<Type...>::value_type v)
{
	dataframe_functors::fill<traits<Type...>::_numCol-1,Type...> filler;
	filler(std::forward<ColumnTuple>(_column_tuple),s,v); 
};

template<class ... Type>
	dataframe<Type...>::dataframe(
		dataframe<Type...>::size_type s)
{
	dataframe_functors::construct<traits<Type...>::_numCol-1,Type...> recurs;
	recurs(std::forward<ColumnTuple>(_column_tuple),s); 
};

template<class ... Type>
	dataframe<Type...>::dataframe(
		dataframe<Type...>::iterator start,
		dataframe<Type...>::iterator stop)
{

	dataframe_functors::it_copy<traits<Type...>::_numCol-1,Type...> it_copier;
	it_copier(
		std::forward<ColumnTuple>(_column_tuple),
		start,
		stop
		); 
};

template<class ... Type>
	dataframe<Type...>::~dataframe(){};

//-------------------container member functions-------------
//-------------------consts

template<class ... Type>
	dataframe<Type...>::reference 
	dataframe<Type...>::at(size_type n)
{
	iterator start=begin(); 
	return *(start+n); 
};

template<class ... Type>
	dataframe<Type...>::reference	
	dataframe<Type...>::front()
{
	return at(0); 
};
template<class ... Type>
	dataframe<Type...>::reference 
	dataframe<Type...>::back()
{
	return at(size()-1);
};
//--------------begin iterators
template<class ... Type>
	dataframe<Type...>::iterator 
	dataframe<Type...>::begin()
{
	iterator it(_column_tuple); 
	return it; 
};
template<class ... Type>
	dataframe<Type...>::const_iterator 
	dataframe<Type...>::begin()const
{
	return cbegin(); 
};
template<class ... Type>
	dataframe<Type...>::const_iterator 
	dataframe<Type...>::cbegin()const
{
	const_iterator it(_column_tuple); 
	return it; 
};
template<class ... Type>
	dataframe<Type...>::reverse_iterator 
	dataframe<Type...>::rbegin()
{
	reverse_iterator it=rend();
	it+=size();  
	return it; 
};
template<class ... Type>
	dataframe<Type...>::const_reverse_iterator 
	dataframe<Type...>::rbegin()const
{
	return crbegin(); 
};
template<class ... Type>
	dataframe<Type...>::const_reverse_iterator 
	dataframe<Type...>::crbegin()const
{
	const_reverse_iterator it=crend();
	it+=size();
	return it; 
};
template<class ... Type>
	dataframe<Type...>::zip_iterator 
	dataframe<Type...>::begin_zip()
{
	zip_iterator it(begin()); 
	return it; 
};
template<class ... Type>
	dataframe<Type...>::const_zip_iterator 
	dataframe<Type...>::begin_zip()const
{
	return cbegin_zip(); 
};
template<class ... Type>
	dataframe<Type...>::const_zip_iterator 
	dataframe<Type...>::cbegin_zip()const
{
	const_zip_iterator it(cbegin()); 
	return it; 
};
//------------------end iterators 
template<class ... Type>
	dataframe<Type...>::iterator 
	dataframe<Type...>::end()
{
	iterator it=begin(); 
	it+=size(); 	
	return it;
}; 
template<class ... Type>
	dataframe<Type...>::const_iterator 
	dataframe<Type...>::end()const
{
	return cend();
}; 
template<class ... Type>
	dataframe<Type...>::const_iterator 
	dataframe<Type...>::cend()const
{
	const_iterator it=cbegin(); 
	it+=size(); 	
	return it;
};
template<class ... Type>
	dataframe<Type...>::reverse_iterator 
	dataframe<Type...>::rend()
{
	reverse_iterator it(_column_tuple); 
	return it; 
};
template<class ... Type>
	dataframe<Type...>::const_reverse_iterator 
	dataframe<Type...>::rend()const
{
	return crbegin(); 
};
template<class ... Type>
	dataframe<Type...>::const_reverse_iterator 
	dataframe<Type...>::crend()const
{
	const_reverse_iterator it(_column_tuple); 
	return it; 
};
template<class ... Type>
	dataframe<Type...>::zip_iterator 
	dataframe<Type...>::end_zip()
{
	zip_iterator it(end()); 
	return it; 
};
template<class ... Type>
	dataframe<Type...>::const_zip_iterator 
	dataframe<Type...>::end_zip()const
{
	return cend_zip(); 
};
template<class ... Type>
	dataframe<Type...>::const_zip_iterator 
	dataframe<Type...>::cend_zip()const
{
	const_zip_iterator it(cend()); 
	return it; 
};

//-----------------------querry members
template<class ... Type>
	dataframe<Type...>::size_type 
	dataframe<Type...>::size()const
{
	typedef typename column_tuple<Type...>::element<0>::type Column;
	
	const Column& column=std::get<0>(_column_tuple);
	size_type size=column.size();
	return size;
};

template<class ... Type>
	dataframe<Type...>::size_type 
	dataframe<Type...>::max_size()const
{
	typedef typename column_tuple<Type...>::element<0>::type Column;
	
	const Column& column=std::get<0>(_column_tuple);
	size_type maxSize=column.max_size();
	return maxSize;
};

template<class ... Type>
	bool 
	dataframe<Type...>::empty()const
{
	typedef typename column_tuple<Type...>::element<0>::type Column;
	
	const Column& column=std::get<0>(_column_tuple);
	return column.empty(); 
};

template<class ... Type>
	void 
	dataframe<Type...>::reserve(dataframe<Type...>::size_type s)
{
	dataframe_functors::	
				reserve<traits<Type...>::_numCol-1,Type...> recursive;
	recursive(std::forward<ColumnTuple>(_column_tuple),s); 
};

template<class ... Type>
	dataframe<Type...>::size_type 
	dataframe<Type...>::capacity()const
{	
	typedef typename column_tuple<Type...>::element<0>::type Column;
	
	const Column& column=std::get<0>(_column_tuple);
	return column.capacity(); 
};

//-------------------non consts
template<class ... Type>
	void 
	dataframe<Type...>::assign(
		dataframe<Type...>::iterator start,
		dataframe<Type...>::iterator stop)
{
	clear(); 
	dataframe_functors::	
				assign_range<traits<Type...>::_numCol-1,Type...> recursive;
	recursive(std::forward<ColumnTuple>(_column_tuple),start,stop); 
};
template<class ... Type>
	void 
	dataframe<Type...>::assign(
		dataframe<Type...>::size_type s,
		dataframe<Type...>::value_type v)
{
	clear(); 
	dataframe_functors::assign_value<traits<Type...>::_numCol-1,Type...> recurs;
	recurs(std::forward<ColumnTuple>(_column_tuple),s,v); 
};
template<class ... Type>
	void 
	dataframe<Type...>::clear()
{
	dataframe_functors::clear<traits<Type...>::_numCol-1,Type...> recursive;
	recursive(std::forward<ColumnTuple>(_column_tuple)); 
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
	recurs(std::forward<ColumnTuple>(_column_tuple),pos,v); 
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
	recurs(	std::forward<ColumnTuple>(_column_tuple),
			pos,
			start,
			stop); 
	return pos; 
};

template<class ... Type>
	dataframe<Type...>::iterator 
	dataframe<Type...>::erase(
		dataframe<Type...>::iterator pos)
{
	dataframe_functors::erase_value<traits<Type...>::_numCol-1,Type...> recurs;
	recurs(std::forward<ColumnTuple>(_column_tuple),pos);
	return pos;  
};

template<class ... Type>
	dataframe<Type...>::iterator 
	dataframe<Type...>::erase(
		dataframe<Type...>::iterator start,
		dataframe<Type...>::iterator stop)
{
	dataframe_functors::erase_range<traits<Type...>::_numCol-1,Type...> recurs;
	recurs(std::forward<ColumnTuple>(_column_tuple),start,stop); 
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
	recurs(std::forward<ColumnTuple>(_column_tuple),n); 
};

template<class ... Type>
	void 
	dataframe<Type...>::resize(
		dataframe<Type...>::size_type n,
		dataframe<Type...>::value_type v)
{
	dataframe_functors::resize_value<traits<Type...>::_numCol-1,Type...> recurs;
	recurs(std::forward<ColumnTuple>(_column_tuple),n,v); 
};

template<class ... Type>
	void 
	dataframe<Type...>::swap(
		dataframe<Type...>& other)
{	
	std::swap(	_column_tuple,
				other._column_tuple); 
}; 
//-------------------------operators

template<class ... Type>
	dataframe<Type...>::reference 
	dataframe<Type...>::operator[](
		dataframe<Type...>::size_type n)
{
	return at(n);
};

template<class ... Type>
	dataframe<Type...>&
	dataframe<Type...>::operator=(
		const dataframe<Type...>& other)
{
	_column_tuple=other._column_tuple;
	return *this; 
};

template<class ... Type>
	bool 
	dataframe<Type...>::operator==(
		const dataframe<Type...>& other)const
{
	return _column_tuple==other._column_tuple;
};

template<class ... Type>
	bool 
	dataframe<Type...>::operator!=(
		const dataframe<Type...>& other)const
{
	return !(*this==other); 	
};












