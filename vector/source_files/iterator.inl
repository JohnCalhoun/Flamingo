//iterator.inl
#include"HashedArrayTree.cu"
#include<cstdlib>
#include<functional>
//***************************CONSTRUCTION?DESTRUCTION***********************
template<typename T, Memory M>
template<typename U>
HashedArrayTree<T,M>::Iterator<U>::Iterator(){};

template<typename T, Memory M>
template<typename U>
HashedArrayTree<T,M>::Iterator<U>::Iterator(
				HashedArrayTree<T,M>::tree& tree,
				HashedArrayTree<T,M>::Iterator<U>& it){
	_cordinate.setTree(tree);
	_cordinate.setDistance(it._cordinate.distance() ); 
};
template<typename T, Memory M>
template<typename U>
HashedArrayTree<T,M>::Iterator<U>::Iterator(HashedArrayTree<T,M>::tree& tree){
	_cordinate.setTree(tree);
};

template<typename T, Memory M>
template<typename U>
HashedArrayTree<T,M>::Iterator<U>::Iterator(const HashedArrayTree<T,M>::Iterator<U>& it ){
	_cordinate=it._cordinate;
};

template<typename T, Memory M>
template<typename U>
HashedArrayTree<T,M>::Iterator<U>::~Iterator(){};

template<typename T,Memory M>
template<typename U>
void HashedArrayTree<T,M>::Iterator<U>::initalize(int x,int y){
	_cordinate.setRow(x);	
	_cordinate.setOffset(y);
}

template<typename T,Memory M>
template<typename U>
void HashedArrayTree<T,M>::Iterator<U>::initalize(int x){
	_cordinate.setDistance(x);
}
template<typename T,Memory M>
template<typename U>
void HashedArrayTree<T,M>::Iterator<U>::initalize(HashedArrayTree<T,M>::Cordinate C){
	_cordinate=C;
}


//***************************CONSTRUCTION?DESTRUCTION***********************
///***************************comparison operators***********************
template<typename T,Memory M>
template<typename U>
HashedArrayTree<T,M>::Iterator<U>& HashedArrayTree<T,M>::Iterator<U>::operator=(const HashedArrayTree<T,M>::Iterator<U>& other){
	this->_cordinate=other._cordinate;
	return *this;
};

template<typename T,Memory M>
template<typename U>
bool HashedArrayTree<T,M>::Iterator<U>::operator==(const HashedArrayTree<T,M>::Iterator<U>& it) const{
	bool cor=this->_cordinate==it._cordinate;
	return (cor);
};

template<typename T,Memory M>
template<typename U>
bool HashedArrayTree<T,M>::Iterator<U>::operator!=(const HashedArrayTree<T,M>::Iterator<U>& it) const{
	return !(*this==it);
};

template<typename T,Memory M>
template<typename U>
bool HashedArrayTree<T,M>::Iterator<U>::operator<(const HashedArrayTree<T,M>::Iterator<U>& it) const{
	return comp(_cordinate,it._cordinate);
}; 

template<typename T,Memory M>
template<typename U>
bool HashedArrayTree<T,M>::Iterator<U>::operator>(const HashedArrayTree<T,M>::Iterator<U>& it) const{
	return *this<it;
}; 

template<typename T,Memory M>
template<typename U>
bool HashedArrayTree<T,M>::Iterator<U>::operator<=(const HashedArrayTree<T,M>::Iterator<U>& it) const{
	return !(*this>it);
}; 

template<typename T,Memory M>
template<typename U>
bool HashedArrayTree<T,M>::Iterator<U>::operator>=(const HashedArrayTree<T,M>::Iterator<U>& it) const{
	return !(*this<it);
}; 

//***************************comparison operators***********************
//***************************arithmic operators**********************

template<typename T,Memory M>
template<typename U>
HashedArrayTree<T,M>::Iterator<U>& HashedArrayTree<T,M>::Iterator<U>::operator++(){
	iterator temp(*this);
	*this=temp+1; 
	return *this; 
};

template<typename T,Memory M>
template<typename U>
HashedArrayTree<T,M>::Iterator<U> HashedArrayTree<T,M>::Iterator<U>::operator++(int){
	iterator temp(*this);
	++(*this);
	return temp; 
}; 

template<typename T,Memory M>
template<typename U>
HashedArrayTree<T,M>::Iterator<U>& HashedArrayTree<T,M>::Iterator<U>::operator--(){
	iterator temp(*this);
	this*=temp-1; 
	return *this; 
}; 

template<typename T,Memory M>
template<typename U>
HashedArrayTree<T,M>::Iterator<U> HashedArrayTree<T,M>::Iterator<U>::operator--(int){
	iterator temp(*this);
	--(*this);
	return temp; 
}; 

template<typename T,Memory M>
template<typename U>
HashedArrayTree<T,M>::Iterator<U> HashedArrayTree<T,M>::Iterator<U>::operator+(HashedArrayTree<T,M>::size_type x) const{
	iterator it(*this);
	int tmp=this->op(	_cordinate.distance(),
					x); 
	Cordinate cor=_cordinate;
	cor.setDistance(tmp);
	it.initalize(cor);

	return it;
}; 

template<typename T,Memory M>
template<typename U>
HashedArrayTree<T,M>::Iterator<U> HashedArrayTree<T,M>::Iterator<U>::operator-(HashedArrayTree<T,M>::size_type x) const{
	iterator it(*this);
	return it+(-x);
};

template<typename T,Memory M>
template<typename U>
HashedArrayTree<T,M>::difference_type HashedArrayTree<T,M>::Iterator<U>::operator-(HashedArrayTree<T,M>::Iterator<U> other) const{
	int this_total=_cordinate.distance();
	int other_total=(other._cordinate).offset();
	return this_total-other_total; 
};

template<typename T,Memory M>
template<typename U>
HashedArrayTree<T,M>::Iterator<U>& HashedArrayTree<T,M>::Iterator<U>::operator+=(HashedArrayTree<T,M>::size_type x){
	iterator temp(*this);
	this*=temp+x; 
	return *this; 
}; 

template<typename T,Memory M>
template<typename U>
HashedArrayTree<T,M>::Iterator<U>& HashedArrayTree<T,M>::Iterator<U>::operator-=(HashedArrayTree<T,M>::size_type x){
	iterator temp(*this);
	this*=temp-x; 
	return *this; 
};
//***************************arithmic operators***********************
//***************************pointer operators***********************

template<typename T,Memory M>
template<typename U>
HashedArrayTree<T,M>::reference HashedArrayTree<T,M>::Iterator<U>::operator*(){
	return reference(_cordinate.access());
};

template<typename T,Memory M>
template<typename U>
HashedArrayTree<T,M>::pointer HashedArrayTree<T,M>::Iterator<U>::operator->(){
	return _cordinate.access();
};

template<typename T,Memory M>
template<typename U>
HashedArrayTree<T,M>::reference HashedArrayTree<T,M>::Iterator<U>::operator[](HashedArrayTree<T,M>::size_type x){
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
