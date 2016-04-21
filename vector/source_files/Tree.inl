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
#include "cordinate.h"
#include <thrust/device_ptr.h>


namespace Flamingo {
namespace Vector {

template<typename T, typename A>
Tree<T,A>::Tree():Tree(0){};

template<typename T, typename A>
Tree<T,A>::Tree(int size){
	resize(size); 
	for(auto it=_root.begin();it<_root.end(); it++){
		*it=NULL; 
	}
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
	_root[_openbranch]=p;
	if(_openbranch<width()){
		_openbranch++; 
	};
};
template<typename T, typename A>
void Tree<T,A>::addbranch(){
	if(_openbranch<width()){

		pointer p=_allocator.allocate(width()*sizeof(T));

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
		if(p){
			_allocator.deallocate(p);
		}
		_root[x]=NULL;
	}
};
template<typename T, typename A>
bool Tree<T,A>::isfree()const{
	return width()>openbranch(); 
};
template<typename T, typename A>
void Tree<T,A>::clear(){
	while(openbranch()>0){
		removebranch(); 		
	}

};
template<typename T, typename A>
size_t Tree<T,A>::size()const{
	return _root.size(); 
};
template<typename T, typename A>
void Tree<T,A>::resize(int y){

	if( width()==0 ){
		_root.resize(y,NULL); 
		setopenbranch(0); 
	}else if( y!=width() )
	{
		if(y > width()) { 
			Tree<T,A> tmp(y); 
			int factor=y/width(); 
			for(int i=0;i<openbranch(); i++){
				int mod		=i%factor;
				int row_into	=i/factor; 

				if(mod==0){
					tmp.addbranch(); 
				}				
				pointer dst=tmp.getbranch(row_into)+width()*mod; 
				Location::MemCopy(	getbranch(i),
								dst,
								width()*sizeof(pointer));
			}
			swap(tmp); 
		}else if(y < (width()-2) ){}
	}
};
template<typename T, typename A>
Tree<T,A>::const_iterator Tree<T,A>::cbegin()const{	
	return const_iterator(thrust::raw_pointer_cast( _root.data() )); 
};
template<typename T, typename A>
Tree<T,A>::const_iterator Tree<T,A>::cend()const{
	return cbegin()+width(); 
};
template<typename T, typename A>
Tree<T,A>::iterator Tree<T,A>::begin(){	
	return thrust::raw_pointer_cast( _root.data() ); 
};
template<typename T, typename A>
Tree<T,A>::iterator Tree<T,A>::end(){
	return begin()+width(); 
};
template<typename T, typename A>
const Tree<T,A>::Root& Tree<T,A>::root()const{
	return _root; 
};
template<typename T, typename A>
Tree<T,A>::Root& Tree<T,A>::root(){
	return _root; 
};
template<typename T, typename A>
template<typename U>
void Tree<T,A>::copy_to_array(U ptr)const{
	for(int i=0; i<openbranch(); i++){
		Location::MemCopy(	getbranch(i), 
						pointer(ptr+i*width()),
						width()*sizeof(T)); 
	}
};

template<typename T, typename A>
template<typename B>
Tree<T,A>& Tree<T,A>::operator=(const Tree<T,B>& other){
	typedef typename Tree<T,B>::const_iterator	other_iterator; 
	typedef typename Tree<T,B>::Root			Other_Root;
	typedef typename Tree<T,B>::pointer		Other_pointer; 

	clear(); 
	resize( other.size() ); 

	while(openbranch()<other.openbranch() ){
		addbranch(); 
	}

	const Other_Root& other_root=other.root(); 
	Root& this_root=this->root(); 

	for(int n=0; n<other.openbranch(); n++){
		pointer src_ptr=Other_pointer(other_root[n]); 
		pointer dst	=this_root[n]; 

		if( src_ptr ){
			Location::MemCopy(	src_ptr,
							dst,
							sizeof(T)*width() 
						); 
		}else{
			this_root[n]=NULL; 
		}
	}	
	return *this;
};
//*****************************non const functions********************
/************************************equality operator******************/
template<typename A,typename B,typename C,typename D>
bool operator==(const Tree<A,B>& tree_1, const Tree<C,D>& tree_2){

	if( tree_1.width()==tree_2.width() ){	
		return tree_1.root()==tree_2.root(); 
	}else{
		return false;
	}
};

template<typename A,typename B>
void Tree<A,B>::swap(Tree<A,B>& other){
	_root.swap(other._root); 	
	std::swap(this->_openbranch,	other._openbranch);
};



}
}



