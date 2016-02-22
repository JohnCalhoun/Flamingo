//iterator.inl
#include "iterator.cpp"
//private functions
template<class ... Type>
dataframe_iterator<Type...>::pointer dataframe_iterator<Type...>::get_pointer() const{
	return _pointer; 
}

//recursive template functors
//nulify
template<int n,typename pointer>
struct nullify {
	void operator()(pointer& p){
		thrust::get<n>(p)=NULL;		
		nullify<n-1,pointer> null_r;
		null_r(p);	
	}
};
template<typename pointer>
struct nullify<0,pointer> {
	void operator()(pointer& p){
		thrust::get<0>(p)=NULL;	
	}
};
//increment
template<int n,typename pointer>
struct increment {
	void operator()(pointer& p){

		pointer tmp=thrust::get<n>(p);
		tmp++;		
		thrust::get<n>(p)=tmp;					
		
		increment<n-1,pointer> inc_r;
		inc_r(p);	
	}
};
template<typename pointer>
struct increment<0,pointer> {
	void operator()(pointer& p){	
		pointer tmp=thrust::get<0>(p);
		tmp++;		
		thrust::get<0>(p)=tmp;				
	}
};
//decrement
template<int n,typename pointer>
struct decrement {
	void operator()(pointer& p){

		pointer tmp=thrust::get<n>(p);
		tmp--;		
		thrust::get<n>(p)=tmp;					
		
		decrement<n-1,pointer> inc_r;
		inc_r(p);	
	}
};
template<typename pointer>
struct decrement<0,pointer> {
	void operator()(pointer& p){	
		pointer tmp=thrust::get<0>(p);
		tmp--;		
		thrust::get<0>(p)=tmp;				
	}
};
//arithmic-plus
template<int n,typename pointer,typename T>
struct arithmic_plus {
	void operator()(pointer& lhs,const T& rhs){

		thrust::get<n>(lhs)+=rhs;
		
		arithmic_plus<n-1,pointer,T> arith_r;
		arith_r(lhs,rhs);	
	}
};
template<typename pointer,typename T>
struct arithmic_plus<0,pointer,T> {
	void operator()(pointer& lhs,const T& rhs){
		thrust::get<0>(lhs)+=rhs;
	}
};
//arithmic-minus
template<int n,typename pointer, typename T>
struct arithmic_minus {
	void operator()(pointer& lhs,const T& rhs){

		thrust::get<n>(lhs)-=rhs;
		
		arithmic_minus<n-1,pointer,T> arith_r;
		arith_r(lhs,rhs);	
	}
};
template<typename pointer,typename T>
struct arithmic_minus<0,pointer,T> {
	void operator()(pointer& lhs,const T& rhs){
		thrust::get<0>(lhs)-=rhs;
	}
};

//derefence 
template<int n,typename pointer, typename value>
struct dereference {
	void operator()(pointer& lhs, value& rhs){

		thrust::get<n>(rhs)=*thrust::get<n>(lhs);
		
		dereference<n-1,pointer,value> deref_r;
		deref_r(lhs,rhs);	
	}
};
template<typename pointer,typename value>
struct dereference <0,pointer,value> {
	void operator()(pointer& lhs,const value& rhs){
		thrust::get<0>(rhs)=*thrust::get<0>(lhs);
	}
};

//public functions
template<class ... Type>
dataframe_iterator<Type...>::dataframe_iterator(){
	nullify<sizeof...(Type),pointer> _null;
	_null(_pointer);
}

template<class ... Type>
dataframe_iterator<Type...>::dataframe_iterator(const dataframe_iterator<Type...>& other){
	_pointer=other.get_pointer();
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
	bool result=thrust::get<0>(_pointer)<thrust::get<0>(other.get_pointer());
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
	increment<sizeof...(Type),pointer> _inc;
	_inc(_pointer);

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
	decrement<sizeof...(Type),pointer> _dec;
	_dec(_pointer);

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
	arithmic_plus<sizeof...(Type),pointer,size_type> _arith;
	_arith(_pointer,n);

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
	arithmic_minus<sizeof...(Type),pointer,size_type> _arith;
	_arith(_pointer,n);

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
	difference_type result=thrust::get<0>(_pointer)-thrust::get<0>(other.get_pointer());
	return result;
} 


template<class ... Type>
dataframe_iterator<Type...>::reference dataframe_iterator<Type...>::operator*(){
	typedef typename traits<Type...>::value_type value_type;
	
	value_type tmp; 
	dereference <sizeof...(Type),pointer,value_type> _null;
	_null(_pointer,tmp);
	return tmp; 

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
	return thrust::get<n>(_pointer); 
}

















