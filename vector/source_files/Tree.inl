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
	return _root[x]; 
};
template<typename T, typename A>
Tree<T,A>::Root Tree<T,A>::getRoot()const{
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
	_root[x]=p;
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
		pointer p=_root[x];
		if(p)
			_allocator.deallocate(p);
		_root[x]=NULL;
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
const Tree<T,A>::Root& Tree<T,A>::root()const{
	return _root; 
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
	return tree_1.root()==tree_2.root(); 
};

template<typename A,typename B>
void Tree<A,B>::swap(Tree<A,B>& other){
	std::swap(this->_root,		other._root);
	std::swap(this->_openbranch,	other._openbranch);
};







