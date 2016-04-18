//iterator.inl
#include"HashedArrayTree.cu"
#include<cstdlib>
#include<functional>
//***************************CONSTRUCTION?DESTRUCTION***********************

template<typename V,typename A,typename U>
Iterator<V,A,U>::Iterator(){};


template<typename V,typename A,typename U>
Iterator<V,A,U>::Iterator(
				Iterator<V,A,U>::tree& tree,
				Iterator<V,A,U>& it)
	:_cordinate(tree,it._cordinate.distance() )
{};

template<typename V,typename A,typename U>
Iterator<V,A,U>::Iterator(
	const Iterator<V,A,U>::tree& tree)
	:_cordinate(tree)
{};


template<typename V,typename A,typename U>
Iterator<V,A,U>::Iterator(const Iterator<V,A,U>& it ){
	_cordinate=it._cordinate;
};


template<typename V,typename A,typename U>
Iterator<V,A,U>::~Iterator(){};


template<typename V,typename A,typename U>
void Iterator<V,A,U>::initalize(Iterator<V,A,U>::size_type x,Iterator<V,A,U>::size_type y){
	_cordinate.set(x,y);	
}


template<typename V,typename A,typename U>
void Iterator<V,A,U>::initalize(Iterator<V,A,U>::size_type x){
	_cordinate.setDistance(x);
}

template<typename V,typename A,typename U>
void Iterator<V,A,U>::initalize(Cordinate C){
	_cordinate=C;
}


//***************************CONSTRUCTION?DESTRUCTION***********************
///***************************comparison operators***********************

template<typename V,typename A,typename U>
Iterator<V,A,U>& Iterator<V,A,U>::operator=(const Iterator<V,A,U>& other){
	this->_cordinate=other._cordinate;
	return *this;
};


template<typename V,typename A,typename U>
bool Iterator<V,A,U>::operator==(const Iterator<V,A,U>& it) const{
	bool cor=this->_cordinate==it._cordinate;
	return (cor);
};


template<typename V,typename A,typename U>
bool Iterator<V,A,U>::operator!=(const Iterator<V,A,U>& it) const{
	return !(*this==it);
};


template<typename V,typename A,typename U>
bool Iterator<V,A,U>::operator<(const Iterator<V,A,U>& it) const{
	return comp(_cordinate,it._cordinate);
}; 


template<typename V,typename A,typename U>
bool Iterator<V,A,U>::operator>(const Iterator<V,A,U>& it) const{
	return *this<it;
}; 


template<typename V,typename A,typename U>
bool Iterator<V,A,U>::operator<=(const Iterator<V,A,U>& it) const{
	return !(*this>it);
}; 


template<typename V,typename A,typename U>
bool Iterator<V,A,U>::operator>=(const Iterator<V,A,U>& it) const{
	return !(*this<it);
}; 

//***************************comparison operators***********************
//***************************arithmic operators**********************


template<typename V,typename A,typename U>
Iterator<V,A,U>& Iterator<V,A,U>::operator++(){
	iterator temp(*this);
	*this=temp+1; 
	return *this; 
};


template<typename V,typename A,typename U>
Iterator<V,A,U> Iterator<V,A,U>::operator++(int){
	iterator temp(*this);
	++(*this);
	return temp; 
}; 


template<typename V,typename A,typename U>
Iterator<V,A,U>& Iterator<V,A,U>::operator--(){
	iterator temp(*this);
	this*=temp-1; 
	return *this; 
}; 


template<typename V,typename A,typename U>
Iterator<V,A,U> Iterator<V,A,U>::operator--(int){
	iterator temp(*this);
	--(*this);
	return temp; 
}; 


template<typename V,typename A,typename U>
Iterator<V,A,U> Iterator<V,A,U>::operator+(Iterator<V,A,U>::size_type x) const{
	iterator it(*this);
	int dist=this->op(	_cordinate.distance(),
					x); 
	it.initalize(dist); 

	return it;
}; 


template<typename V,typename A,typename U>
Iterator<V,A,U> Iterator<V,A,U>::operator-(Iterator<V,A,U>::size_type x) const{
	iterator it(*this);
	return it+(-x);
};


template<typename V,typename A,typename U>
Iterator<V,A,U>::difference_type Iterator<V,A,U>::operator-(Iterator<V,A,U> other) const{
	int this_total=_cordinate.distance();
	int other_total=(other._cordinate).offset();
	return this_total-other_total; 
};


template<typename V,typename A,typename U>
Iterator<V,A,U>& Iterator<V,A,U>::operator+=(Iterator<V,A,U>::size_type x){
	iterator temp(*this);
	this*=temp+x; 
	return *this; 
}; 


template<typename V,typename A,typename U>
Iterator<V,A,U>& Iterator<V,A,U>::operator-=(Iterator<V,A,U>::size_type x){
	iterator temp(*this);
	this*=temp-x; 
	return *this; 
};
//***************************arithmic operators***********************
//***************************pointer operators***********************


template<typename V,typename A,typename U>
Iterator<V,A,U>::reference Iterator<V,A,U>::operator*(){
	auto tmp=_cordinate.access(); 
	return reference(tmp);
};


template<typename V,typename A,typename U>
Iterator<V,A,U>::pointer Iterator<V,A,U>::operator->(){
	return _cordinate.access();
};


template<typename V,typename A,typename U>
Iterator<V,A,U>::reference Iterator<V,A,U>::operator[](Iterator<V,A,U>::size_type x){
	Cordinate tmp=_cordinate; 
	tmp.setdistance(tmp.distance()+x);
	return *_cordinate.access(); 
};

//***************************pointer operators***********************
//
//
//
//
//
//
//
