//iterator.inl
#include"HashedArrayTree.cu"
#include<cstdlib>
#include<functional>
//***************************CONSTRUCTION?DESTRUCTION***********************
template<typename T, typename L>
template<typename U>
HashedArrayTree<T,L>::Iterator<U>::Iterator(){
	_tree_ptr=NULL;
	_cordinate.setTree(_tree_ptr);
};

template<typename T, typename L>
template<typename U>
HashedArrayTree<T,L>::Iterator<U>::Iterator(HashedArrayTree<T,L>::tree& tree){
	_tree_ptr=&tree; 
	_cordinate.setTree(_tree_ptr);
};

template<typename T, typename L>
template<typename U>
HashedArrayTree<T,L>::Iterator<U>::Iterator(const HashedArrayTree<T,L>::Iterator<U>& it ){
	_tree_ptr=it._tree_ptr; 
	_cordinate.setTree(_tree_ptr);
};

template<typename T, typename L>
template<typename U>
HashedArrayTree<T,L>::Iterator<U>::~Iterator(){};

template<typename T,typename L>
template<typename U>
void HashedArrayTree<T,L>::Iterator<U>::initalize(int x,int y){
	_cordinate.setRow(x);	
	_cordinate.setOffset(y);
}

template<typename T,typename L>
template<typename U>
void HashedArrayTree<T,L>::Iterator<U>::initalize(int x){
	_cordinate.setDistance(x);
}
template<typename T,typename L>
template<typename U>
void HashedArrayTree<T,L>::Iterator<U>::initalize(HashedArrayTree<T,L>::Cordinate C){
	_cordinate=C;
}


//***************************CONSTRUCTION?DESTRUCTION***********************
///***************************comparison operators***********************
template<typename T,typename L>
template<typename U>
HashedArrayTree<T,L>::Iterator<U>& HashedArrayTree<T,L>::Iterator<U>::operator=(const HashedArrayTree<T,L>::Iterator<U>& other){
	this->_cordinate=other._cordinate;
	this->_tree_ptr=other._tree_ptr; 
	return *this;
};

template<typename T,typename L>
template<typename U>
bool HashedArrayTree<T,L>::Iterator<U>::operator==(const HashedArrayTree<T,L>::Iterator<U>& it) const{
	bool cor=this->_cordinate==it._cordinate;
	bool	tre=this->_tree_ptr==it._tree_ptr; 
	return (cor and tre);
};

template<typename T,typename L>
template<typename U>
bool HashedArrayTree<T,L>::Iterator<U>::operator!=(const HashedArrayTree<T,L>::Iterator<U>& it) const{
	return !(*this==it);
};

template<typename T,typename L>
template<typename U>
bool HashedArrayTree<T,L>::Iterator<U>::operator<(const HashedArrayTree<T,L>::Iterator<U>& it) const{
	return comp(_cordinate,it._cordinate);
}; 

template<typename T,typename L>
template<typename U>
bool HashedArrayTree<T,L>::Iterator<U>::operator>(const HashedArrayTree<T,L>::Iterator<U>& it) const{
	return *this<it;
}; 

template<typename T,typename L>
template<typename U>
bool HashedArrayTree<T,L>::Iterator<U>::operator<=(const HashedArrayTree<T,L>::Iterator<U>& it) const{
	return !(*this>it);
}; 

template<typename T,typename L>
template<typename U>
bool HashedArrayTree<T,L>::Iterator<U>::operator>=(const HashedArrayTree<T,L>::Iterator<U>& it) const{
	return !(*this<it);
}; 

//***************************comparison operators***********************
//***************************arithmic operators**********************

template<typename T,typename L>
template<typename U>
HashedArrayTree<T,L>::Iterator<U>& HashedArrayTree<T,L>::Iterator<U>::operator++(){
	iterator temp(*this);
	*this=temp+1; 
	return *this; 
};

template<typename T,typename L>
template<typename U>
HashedArrayTree<T,L>::Iterator<U> HashedArrayTree<T,L>::Iterator<U>::operator++(int){
	iterator temp(*this);
	++(*this);
	return temp; 
}; 

template<typename T,typename L>
template<typename U>
HashedArrayTree<T,L>::Iterator<U>& HashedArrayTree<T,L>::Iterator<U>::operator--(){
	iterator temp(*this);
	this*=temp-1; 
	return *this; 
}; 

template<typename T,typename L>
template<typename U>
HashedArrayTree<T,L>::Iterator<U> HashedArrayTree<T,L>::Iterator<U>::operator--(int){
	iterator temp(*this);
	--(*this);
	return temp; 
}; 

template<typename T,typename L>
template<typename U>
HashedArrayTree<T,L>::Iterator<U> HashedArrayTree<T,L>::Iterator<U>::operator+(HashedArrayTree<T,L>::size_type x) const{
	iterator it(*this);
	int tmp=this->op(	_cordinate.distance(),
					x); 
	Cordinate cor=_cordinate;
	cor.setDistance(tmp);
	it.initalize(cor);

	return it;
}; 

template<typename T,typename L>
template<typename U>
HashedArrayTree<T,L>::Iterator<U> HashedArrayTree<T,L>::Iterator<U>::operator-(HashedArrayTree<T,L>::size_type x) const{
	iterator it(*_tree_ptr);
	return it+(-x);
};

template<typename T,typename L>
template<typename U>
HashedArrayTree<T,L>::difference_type HashedArrayTree<T,L>::Iterator<U>::operator-(HashedArrayTree<T,L>::Iterator<U> other) const{
	int this_total=_cordinate.distance();
	int other_total=(other._cordinate).offset();
	return this_total-other_total; 
};

template<typename T,typename L>
template<typename U>
HashedArrayTree<T,L>::Iterator<U>& HashedArrayTree<T,L>::Iterator<U>::operator+=(HashedArrayTree<T,L>::size_type x){
	iterator temp(*this);
	this*=temp+x; 
	return *this; 
}; 

template<typename T,typename L>
template<typename U>
HashedArrayTree<T,L>::Iterator<U>& HashedArrayTree<T,L>::Iterator<U>::operator-=(HashedArrayTree<T,L>::size_type x){
	iterator temp(*this);
	this*=temp-x; 
	return *this; 
};
//***************************arithmic operators***********************
//***************************pointer operators***********************

template<typename T,typename L>
template<typename U>
HashedArrayTree<T,L>::reference HashedArrayTree<T,L>::Iterator<U>::operator*(){
	return *_cordinate.access();
};

template<typename T,typename L>
template<typename U>
HashedArrayTree<T,L>::pointer HashedArrayTree<T,L>::Iterator<U>::operator->(){
	return _cordinate.access();
};

template<typename T,typename L>
template<typename U>
HashedArrayTree<T,L>::reference HashedArrayTree<T,L>::Iterator<U>::operator[](HashedArrayTree<T,L>::size_type x){
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
