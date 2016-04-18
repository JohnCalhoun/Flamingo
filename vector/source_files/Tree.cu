#ifndef TREE_H
#define TREE_H
#include<location.cu>
#include<cuda.h>
#include<vector>
#include "traits.cpp"

#define __both__ __device__ __host__ 

namespace Flamingo {
namespace Vector {

template<typename T, typename A>
class Tree {
	public:
	typedef A	Allocator;

	typedef typename Allocator::Location_Policy	Location;
	typedef T*							pointer;
	typedef typename Allocator::const_pointer	const_pointer;
	typedef typename Allocator::size_type		size_type;
	typedef typename 
		Root_vector<pointer,Location::memory>::type		Root;

	typedef pointer*						iterator; 
	typedef pointer*						const_iterator; 

	typedef T								value_type;
	typedef int							Width;
	//constructors
	Tree();
	Tree(int);
	~Tree();

	//functions
	Width			width()const;
	int				openbranch()const; 
	pointer			getbranch(const int)const;
	Root				getRoot()const;
	bool				isfree()const;
	size_t			size()const;

	void		setopenbranch(int);
	void		addbranch();
	void		addbranch(pointer);
	void		removebranch();
	void		removebranch(int);
	void		replacebranch(pointer,int);
	void		clear();
	void		resize(int); 
	const Root&	root()const;
	Root&	root(); 

	template<typename U>
	void copy_to_array(U)const; 

	iterator begin();
	const_iterator begin()const{return cbegin(); }; 
	const_iterator cbegin()const;

	iterator end(); 
	const_iterator bend()const{return cend(); }; 
	const_iterator cend()const;
	
	template<typename B>
	Tree& operator= (const Tree<T,B>&);  
	void swap( Tree<T,A>&);

	private:
	Allocator				_allocator; 
	Root					_root;
	int					_openbranch;
};

template<typename A,typename B, typename C, typename D>
bool operator==(const Tree<A,B>&,const Tree<A,B>&);

}
}

#undef __both__
#include"Tree.inl"

#endif




