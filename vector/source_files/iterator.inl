//iterator.inl
#include"HashedArrayTree.cu"
#include<cstdlib>
#include<functional>
//***************************CONSTRUCTION?DESTRUCTION***********************

template<typename V,typename U>
Iterator<V,U>::Iterator(){};


template<typename V,typename U>
template<typename A>
Iterator<V,U>::Iterator(
				Tree<V,A>& tree,
				Iterator<V,U>& it)
	:_cordinate(tree,it._cordinate.distance() )
{};

template<typename V,typename U>
template<typename A>
Iterator<V,U>::Iterator(
	const Tree<V,A>& tree)
	:_cordinate(tree)
{};


template<typename V,typename U>
Iterator<V,U>::Iterator(const Iterator<V,U>& it ){
	_cordinate=it._cordinate;
};


template<typename V,typename U>
Iterator<V,U>::~Iterator(){};


template<typename V,typename U>
void Iterator<V,U>::initalize(Iterator<V,U>::size_type x,Iterator<V,U>::size_type y){
	_cordinate.set(x,y);	
}


template<typename V,typename U>
void Iterator<V,U>::initalize(Iterator<V,U>::size_type x){
	_cordinate.setDistance(x);
}

template<typename V,typename U>
void Iterator<V,U>::initalize(Cordinate C){
	_cordinate=C;
}


//***************************CONSTRUCTION?DESTRUCTION***********************
///***************************comparison operators***********************

template<typename V,typename U>
Iterator<V,U>& Iterator<V,U>::operator=(const Iterator<V,U>& other){
	this->_cordinate=other._cordinate;
	return *this;
};


template<typename V,typename U>
bool Iterator<V,U>::operator==(const Iterator<V,U>& it) const{
	bool cor=this->_cordinate==it._cordinate;
	return (cor);
};


template<typename V,typename U>
bool Iterator<V,U>::operator!=(const Iterator<V,U>& it) const{
	return !(*this==it);
};


template<typename V,typename U>
bool Iterator<V,U>::operator<(const Iterator<V,U>& it) const{
	U op; 
	return op.comp(_cordinate,it._cordinate);
}; 


template<typename V,typename U>
bool Iterator<V,U>::operator>(const Iterator<V,U>& it) const{
	return *this<it;
}; 


template<typename V,typename U>
bool Iterator<V,U>::operator<=(const Iterator<V,U>& it) const{
	return !(*this>it);
}; 


template<typename V,typename U>
bool Iterator<V,U>::operator>=(const Iterator<V,U>& it) const{
	return !(*this<it);
}; 

//***************************comparison operators***********************
//***************************arithmic operators**********************


template<typename V,typename U>
Iterator<V,U>& Iterator<V,U>::operator++(){
	iterator temp(*this);
	*this=temp+1; 
	return *this; 
};


template<typename V,typename U>
Iterator<V,U> Iterator<V,U>::operator++(int){
	iterator temp(*this);
	++(*this);
	return temp; 
}; 


template<typename V,typename U>
Iterator<V,U>& Iterator<V,U>::operator--(){
	iterator temp(*this);
	*this =temp-1; 
	return *this; 
}; 


template<typename V,typename U>
Iterator<V,U> Iterator<V,U>::operator--(int){
	iterator temp(*this);
	--(*this);
	return temp; 
}; 


template<typename V,typename U>
Iterator<V,U> Iterator<V,U>::operator+(Iterator<V,U>::size_type x) const{
	iterator it(*this);
	int dist=this->op(	_cordinate.distance(),
					x); 
	it.initalize(dist); 

	return it;
}; 


template<typename V,typename U>
Iterator<V,U> Iterator<V,U>::operator-(Iterator<V,U>::size_type x) const{
	iterator it(*this);
	return it+(-x);
};


template<typename V,typename U>
Iterator<V,U>::difference_type Iterator<V,U>::operator-(Iterator<V,U> other) const{
	int this_total=_cordinate.distance();
	int other_total=(other._cordinate).offset();
	return this_total-other_total; 
};


template<typename V,typename U>
Iterator<V,U>& Iterator<V,U>::operator+=(Iterator<V,U>::size_type x){
	iterator temp(*this);
	*this = temp+x; 
	return *this; 
}; 


template<typename V,typename U>
Iterator<V,U>& Iterator<V,U>::operator-=(Iterator<V,U>::size_type x){
	iterator temp(*this);
	*this=temp-x; 
	return *this; 
};
//***************************arithmic operators***********************
//***************************pointer operators***********************


template<typename V,typename U>
Iterator<V,U>::reference Iterator<V,U>::operator*(){
	auto tmp=_cordinate.access(); 
	return reference(tmp);
};


template<typename V,typename U>
Iterator<V,U>::pointer Iterator<V,U>::operator->(){
	return _cordinate.access();
};


template<typename V,typename U>
Iterator<V,U>::reference Iterator<V,U>::operator[](Iterator<V,U>::size_type x){
	Cordinate tmp=_cordinate; 
	tmp.setDistance(tmp.distance()+x);
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
