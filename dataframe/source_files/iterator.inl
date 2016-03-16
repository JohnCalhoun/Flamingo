//iterator.inl
#include "iterator.cpp"
#include "iterator_functors.cpp"
#include "functional"
#include <iostream>
//private functions
template<class ... Type>
dataframe_iterator<Type...>::pointer dataframe_iterator<Type...>::get_pointer() const{
	return _pointer; 
}

//public functions
template<class ... Type>
dataframe_iterator<Type...>::dataframe_iterator(){
	typename iterator_functors::nullify<sizeof...(Type)-1,pointer> _null;
	_null(std::forward<pointer>(_pointer));
}

template<class ... Type>
dataframe_iterator<Type...>::dataframe_iterator(const dataframe_iterator<Type...>& other){
	_pointer=other.get_pointer();
}

template<class ... Type>
dataframe_iterator<Type...>::dataframe_iterator(
	dataframe_iterator<Type...>::ColumnTuple& tuple)
{
	typename iterator_functors::assign<traits<Type...>::_numCol-1,Type...> assigner; 
	assigner(
		std::forward<ColumnTuple>(tuple),
		std::forward<pointer>(_pointer)
	); 
}

template<class ... Type>
dataframe_iterator<Type...>::~dataframe_iterator(){}

template<class ... Type>
dataframe_iterator<Type...>& dataframe_iterator<Type...>::operator=(const dataframe_iterator<Type...>& other){
	swap(other);
	return *this;
}

template<class ... Type>
bool dataframe_iterator<Type...>::operator==(const dataframe_iterator<Type...>& other) const{
	bool result=_pointer==other.get_pointer(); 
	return result;
}

template<class ... Type>
bool dataframe_iterator<Type...>::operator!=(const dataframe_iterator<Type...>& other) const{
	bool result=!(*this==other);
	return result;
}

template<class ... Type>
bool dataframe_iterator<Type...>::operator<(const dataframe_iterator<Type...>& other) const{
	bool result=_pointer<other.get_pointer();
	return result;
} 

template<class ... Type>
bool dataframe_iterator<Type...>::operator>(const dataframe_iterator<Type...>& other) const{
	return (other < *this);
} 

template<class ... Type>
bool dataframe_iterator<Type...>::operator<=(const dataframe_iterator<Type...>& other) const{
	return !(*this > other);
} 

template<class ... Type>
bool dataframe_iterator<Type...>::operator>=(const dataframe_iterator<Type...>& other) const{
	return !(*this < other);
} 

template<class ... Type>
dataframe_iterator<Type...>& dataframe_iterator<Type...>::operator++(){
	typename iterator_functors::increment<sizeof...(Type)-1,pointer> _inc;
	_inc(std::forward<pointer>(_pointer));

	return *this;
}

template<class ... Type>
dataframe_iterator<Type...> dataframe_iterator<Type...>::operator++(int){
	dataframe_iterator<Type...> tmp(*this);
	operator++();
	return tmp;
} 

template<class ... Type>
dataframe_iterator<Type...>& dataframe_iterator<Type...>::operator--(){
	typename iterator_functors::decrement<sizeof...(Type)-1,pointer> _dec;
	_dec(std::forward<pointer>(_pointer));

	return *this;
} 

template<class ... Type>
dataframe_iterator<Type...> dataframe_iterator<Type...>::operator--(int){
	dataframe_iterator<Type...> tmp(*this);
	operator--();
	return tmp;
} 

template<class ... Type>
dataframe_iterator<Type...>& dataframe_iterator<Type...>::operator+=(dataframe_iterator<Type...>::size_type n){
	typename iterator_functors::arithmic_plus<sizeof...(Type)-1,pointer,size_type> _arith;

	_arith(std::forward<pointer>(_pointer),n);
	return *this;
} 

template<class ... Type>
dataframe_iterator<Type...> dataframe_iterator<Type...>::operator+(dataframe_iterator<Type...>::size_type n) const{
	dataframe_iterator<Type...> tmp(*this);
	tmp+=n;
	return tmp;
} 

template<class ... Type>
dataframe_iterator<Type...>& dataframe_iterator<Type...>::operator-=(dataframe_iterator<Type...>::size_type n){
	typename iterator_functors::arithmic_minus<sizeof...(Type)-1,pointer,size_type> _arith;
	_arith(std::forward<pointer>(_pointer),n);

	return *this;
}  

template<class ... Type>
dataframe_iterator<Type...> dataframe_iterator<Type...>::operator-(dataframe_iterator<Type...>::size_type n) const{
	dataframe_iterator<Type...> tmp(*this);
	tmp-=n;
	return tmp;
} 

template<class ... Type>
dataframe_iterator<Type...>::difference_type dataframe_iterator<Type...>:: operator-(const dataframe_iterator<Type...>& other) const{
	difference_type result=std::get<0>(_pointer)-std::get<0>(other.get_pointer());
	return result;
} 


template<class ... Type>
dataframe_iterator<Type...>::reference dataframe_iterator<Type...>::operator*(){
	
	iterator_functors::
		dereference<sizeof...(Type)-1,Type...> rec;
	return rec(std::forward<pointer>(_pointer));
}

template<class ... Type>
dataframe_iterator<Type...>::reference dataframe_iterator<Type...>::operator[](dataframe_iterator<Type...>::size_type n){
	typedef typename traits<Type...>::value_type value_type;

	value_type tmp(*this); 
	tmp+=n;
	return *tmp;
}

template<class ... Type>
template<int n>
typename traits<Type...>::Return<n>::pointer_base dataframe_iterator<Type...>::get(){
	return  std::get<n>(_pointer); 
}

















