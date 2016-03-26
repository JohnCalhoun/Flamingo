//iterator.inl
#include "iterator.cpp"
#include "iterator_functors.cpp"
#include "functional"
#include <iostream>
//private functions
template<typename ref_type,typename pointer_type,class ... Type>
dataframe_iterator<ref_type,pointer_type,Type...>::pointer dataframe_iterator<ref_type,pointer_type,Type...>::get_pointer() const{
	return _pointer; 
}

//public functions
template<typename ref_type,typename pointer_type,class ... Type>
dataframe_iterator<ref_type,pointer_type,Type...>::dataframe_iterator(){
	typename iterator_functors::nullify<sizeof...(Type)-1,pointer> _null;
	_null(std::forward<pointer>(_pointer));
}

template<typename ref_type,typename pointer_type,class ... Type>
dataframe_iterator<ref_type,pointer_type,Type...>::dataframe_iterator(const dataframe_iterator<ref_type,pointer_type,Type...>& other){
	_pointer=other.get_pointer();
}

template<typename ref_type,typename pointer_type,class ... Type>
dataframe_iterator<ref_type,pointer_type,Type...>::dataframe_iterator(
	dataframe_iterator<ref_type,pointer_type,Type...>::ColumnTuple& tuple)
{
	typename iterator_functors::assign<traits<Type...>::_numCol-1,Type...> assigner; 
	assigner(
		std::forward<ColumnTuple>(tuple),
		std::forward<pointer>(_pointer)
	); 
}
template<typename ref_type,typename pointer_type,class ... Type>
dataframe_iterator<ref_type,pointer_type,Type...>::dataframe_iterator(
	const dataframe_iterator<ref_type,pointer_type,Type...>::ColumnTuple& tuple)
{
	typename iterator_functors::assign_const<traits<Type...>::_numCol-1,Type...> assigner; 
	assigner(
		std::forward<const ColumnTuple>(tuple),
		std::forward<pointer>(_pointer)
	); 
}

template<typename ref_type,typename pointer_type,class ... Type>
dataframe_iterator<ref_type,pointer_type,Type...>::~dataframe_iterator(){}

template<typename ref_type,typename pointer_type,class ... Type>
dataframe_iterator<ref_type,pointer_type,Type...>& dataframe_iterator<ref_type,pointer_type,Type...>::operator=( dataframe_iterator<ref_type,pointer_type,Type...> other){
	swap(other);
	return *this;
}

template<typename ref_type,typename pointer_type,class ... Type>
bool dataframe_iterator<ref_type,pointer_type,Type...>::operator==(const dataframe_iterator<ref_type,pointer_type,Type...>& other) const{
	bool result=_pointer==other.get_pointer(); 
	return result;
}

template<typename ref_type,typename pointer_type,class ... Type>
bool dataframe_iterator<ref_type,pointer_type,Type...>::operator!=(const dataframe_iterator<ref_type,pointer_type,Type...>& other) const{
	bool result=!(*this==other);
	return result;
}

template<typename ref_type,typename pointer_type,class ... Type>
bool dataframe_iterator<ref_type,pointer_type,Type...>::operator<(const dataframe_iterator<ref_type,pointer_type,Type...>& other) const{
	bool result=_pointer<other.get_pointer();
	return result;
} 

template<typename ref_type,typename pointer_type,class ... Type>
bool dataframe_iterator<ref_type,pointer_type,Type...>::operator>(const dataframe_iterator<ref_type,pointer_type,Type...>& other) const{
	return (other < *this);
} 

template<typename ref_type,typename pointer_type,class ... Type>
bool dataframe_iterator<ref_type,pointer_type,Type...>::operator<=(const dataframe_iterator<ref_type,pointer_type,Type...>& other) const{
	return !(*this > other);
} 

template<typename ref_type,typename pointer_type,class ... Type>
bool dataframe_iterator<ref_type,pointer_type,Type...>::operator>=(const dataframe_iterator<ref_type,pointer_type,Type...>& other) const{
	return !(*this < other);
} 

template<typename ref_type,typename pointer_type,class ... Type>
dataframe_iterator<ref_type,pointer_type,Type...>& dataframe_iterator<ref_type,pointer_type,Type...>::operator++(){
	typename iterator_functors::increment<sizeof...(Type)-1,pointer> _inc;
	_inc(std::forward<pointer>(_pointer));

	return *this;
}

template<typename ref_type,typename pointer_type,class ... Type>
dataframe_iterator<ref_type,pointer_type,Type...> dataframe_iterator<ref_type,pointer_type,Type...>::operator++(int){
	dataframe_iterator<ref_type,pointer_type,Type...> tmp(*this);
	operator++();
	return tmp;
} 

template<typename ref_type,typename pointer_type,class ... Type>
dataframe_iterator<ref_type,pointer_type,Type...>& dataframe_iterator<ref_type,pointer_type,Type...>::operator--(){
	typename iterator_functors::decrement<sizeof...(Type)-1,pointer> _dec;
	_dec(std::forward<pointer>(_pointer));

	return *this;
} 

template<typename ref_type,typename pointer_type,class ... Type>
dataframe_iterator<ref_type,pointer_type,Type...> dataframe_iterator<ref_type,pointer_type,Type...>::operator--(int){
	dataframe_iterator<ref_type,pointer_type,Type...> tmp(*this);
	operator--();
	return tmp;
} 

template<typename ref_type,typename pointer_type,class ... Type>
dataframe_iterator<ref_type,pointer_type,Type...>& dataframe_iterator<ref_type,pointer_type,Type...>::operator+=(dataframe_iterator<ref_type,pointer_type,Type...>::size_type n){
	typename iterator_functors::arithmic_plus<sizeof...(Type)-1,pointer,size_type> _arith;

	_arith(std::forward<pointer>(_pointer),n);
	return *this;
} 

template<typename ref_type,typename pointer_type,class ... Type>
dataframe_iterator<ref_type,pointer_type,Type...> dataframe_iterator<ref_type,pointer_type,Type...>::operator+(dataframe_iterator<ref_type,pointer_type,Type...>::size_type n) const{
	dataframe_iterator<ref_type,pointer_type,Type...> tmp(*this);
	tmp+=n;
	return tmp;
} 

template<typename ref_type,typename pointer_type,class ... Type>
dataframe_iterator<ref_type,pointer_type,Type...>& dataframe_iterator<ref_type,pointer_type,Type...>::operator-=(dataframe_iterator<ref_type,pointer_type,Type...>::size_type n){
	typename iterator_functors::arithmic_minus<sizeof...(Type)-1,pointer,size_type> _arith;
	_arith(std::forward<pointer>(_pointer),n);

	return *this;
}  

template<typename ref_type,typename pointer_type,class ... Type>
dataframe_iterator<ref_type,pointer_type,Type...> dataframe_iterator<ref_type,pointer_type,Type...>::operator-(dataframe_iterator<ref_type,pointer_type,Type...>::size_type n) const{
	dataframe_iterator<ref_type,pointer_type,Type...> tmp(*this);
	tmp-=n;
	return tmp;
} 

template<typename ref_type,typename pointer_type,class ... Type>
dataframe_iterator<ref_type,pointer_type,Type...>::difference_type dataframe_iterator<ref_type,pointer_type,Type...>:: operator-(const dataframe_iterator<ref_type,pointer_type,Type...>& other) const{
	difference_type result=std::get<0>(_pointer)-std::get<0>(other.get_pointer());
	return result;
} 

template<typename ref_type,typename pointer_type,class ... Type>
dataframe_iterator<ref_type,pointer_type,Type...>::reference dataframe_iterator<ref_type,pointer_type,Type...>::operator*(){
	
	iterator_functors::
		dereference<sizeof...(Type)-1,reference,pointer,Type...> rec;
	return rec(std::forward<pointer>(_pointer));
}

template<typename ref_type,typename pointer_type,class ... Type>
dataframe_iterator<ref_type,pointer_type,Type...>::reference dataframe_iterator<ref_type,pointer_type,Type...>::operator[](dataframe_iterator<ref_type,pointer_type,Type...>::size_type n){
	return *(this+n);
}

template<typename ref_type,typename pointer_type,class ... Type>
template<int n>
typename traits<Type...>::Return<n>::pointer_base dataframe_iterator<ref_type,pointer_type,Type...>::get(){
	return  std::get<n>(_pointer); 
}

template<typename ref_type,typename pointer_type,class ... Type>
void dataframe_iterator<ref_type,pointer_type,Type...>::swap(dataframe_iterator<ref_type,pointer_type,Type...>& other){
	std::swap(_pointer,other._pointer);
}

template<typename ref_type,typename pointer_type,class ... Type>
dataframe_iterator<ref_type,pointer_type,Type...>::operator bool(){
	return  std::get<0>(_pointer); 
}















