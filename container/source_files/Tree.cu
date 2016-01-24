#ifndef TREE_H
#define TREE_H
#include<location.cu>
#include<cuda.h>
#include"root.h"

#define __both__ __device__ __host__ 

template<typename T, typename A>
class Tree {
	public:
	typedef A	allocator_type;

	typedef typename allocator_type::location	location;
	typedef Root<T,allocator_type,location>		root;
	typedef typename root::Pointer_dev			pointer; 

	//other
	typedef T								value_type;
	typedef int							Width;
	//data
	private:
	allocator_type			_allocator; 
	root					_root;
	int					_openbranch;
	public: 	
	public: 
	//constructors
	Tree();
	Tree(int);
	~Tree();

	//functions
	__both__  Width	width()const;
	int				openbranch()const; 
	__both__  pointer	getbranch(const int)const;
	root				getRoot()const;
	bool				isfree()const;

	void		setopenbranch(int);
	void		addbranch();
	void		addbranch(pointer);
	void		removebranch();
	void		removebranch(int);
	void		replacebranch(pointer,int);
	void		replaceRoot(root); 
	void		clear();
	void		resize(int); 

	Tree& operator= (const Tree&);  
	void swap( Tree<T,A>&);
};

template<typename A,typename B, typename C, typename D>
bool operator==(const Tree<A,B>&,const Tree<A,B>&);


#undef __both__
#include"Tree.inl"
#endif




