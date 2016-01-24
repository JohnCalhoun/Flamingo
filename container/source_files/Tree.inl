#include"Tree.cu"
#include<algorithm>
#include<cuda.h>
#include<cuda_occupancy.h>
#include<iostream>
#include<location.cu>
#include<type_traits>
#include<algorithm>
#include<stdio.h>
#include"internal.h"

template<typename T, typename A>
Tree<T,A>::Tree(){
	resize(0);
	setopenbranch(0);
};

template<typename T, typename A>
Tree<T,A>::Tree(int size){
	setopenbranch(0);
	resize(size); 
};
//*******************************const  functions*********************
template<typename T, typename A>
Tree<T,A>::~Tree(){
};
template<typename T, typename A>
int	Tree<T,A>::width()const{
	return _root.size(); 
};
template<typename T, typename A>
int	Tree<T,A>::openbranch()const{
	return _openbranch; 
};

template<typename T, typename A>
Tree<T,A>::pointer Tree<T,A>::getbranch(const int x)const{
	return _root.get(x); 
};
template<typename T, typename A>
Tree<T,A>::root Tree<T,A>::getRoot()const{
	return _root; 
};
//******************************const functions************************
//******************************non const functions********************
template<typename T, typename A>
void Tree<T,A>::setopenbranch(int x){
	_openbranch=x; 
};

template<typename T, typename A>
void Tree<T,A>::replacebranch(Tree<T,A>::pointer p,int x){
	_root.set(p,x);
};

template<typename T, typename A>
void Tree<T,A>::replaceRoot(Tree<T,A>::root p){
	_root=p; 
};

template<typename T, typename A>
void Tree<T,A>::addbranch(Tree<T,A>::pointer p){
	int offset=_openbranch;
	replacebranch(p,offset); 
	if(_openbranch<width()){
		_openbranch++; 
	};
};
template<typename T, typename A>
void Tree<T,A>::addbranch(){
	if(_openbranch<width()){
		pointer p=_allocator.allocate(width());
		addbranch(p);
	};
};
template<typename T, typename A>
void Tree<T,A>::removebranch(){
		int offset=_openbranch-1;
		removebranch(offset); 
		_openbranch--;
};
template<typename T, typename A>
void Tree<T,A>::removebranch(int x){
	if(x>0 && x<width()){
		pointer p=_root.get(x);
		if(p)
			_allocator.deallocate(p);
		_root.set(NULL,x);
		sync();
	}
};
template<typename T, typename A>
bool Tree<T,A>::isfree()const{
	return width()>openbranch(); 
};
template<typename T, typename A>
void Tree<T,A>::clear(){
	_root.clear(); 
	setopenbranch(0);
};
template<typename T, typename A>
void Tree<T,A>::resize(int y){
	_root.resize(y); 
};
template<typename T, typename A>
Tree<T,A>& Tree<T,A>::operator=(const Tree<T,A>& other){
	_root=other._root;
	_openbranch=other._openbranch;
	return *this;
};
//*****************************non const functions********************
/************************************equality operator******************/
template<typename A,typename B,typename C,typename D>
bool operator==(const Tree<A,B>& tree_1, const Tree<C,D>& tree_2){
	typedef typename B::Location_Policy Location; 
	typedef location<host>			host_location;

	typedef typename std::conditional<	std::is_same<Location,host_location >::value,
							Internal::Equality_host<A,B>,
							Internal::Equality_device<A,B>
						>::type	equality_type;
	
 	typedef typename std::conditional<	std::is_same<A,C>::value,
							equality_type,
							Internal::Equality_false<A,B,C,D>
						>::type	equality_1;
 	typedef typename std::conditional<	std::is_same<B,D>::value,
							equality_1,
							Internal::Equality_false<A,B,C,D>
						>::type	equality;
	
	equality Equality;
	return Equality(tree_1,tree_2); 
};

template<typename A,typename B>
void Tree<A,B>::swap(Tree<A,B>& other){
	std::swap(this->_root,		other._root);
	std::swap(this->_openbranch,	other._openbranch);
};







