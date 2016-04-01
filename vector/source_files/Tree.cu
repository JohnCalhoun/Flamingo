#ifndef TREE_H
#define TREE_H
#include<location.cu>
#include<cuda.h>
#include<vector>

#define __both__ __device__ __host__ 

template<typename T, typename A>
class Tree {
	public:
	typedef A	Allocator;

	typedef typename Allocator::Location_Policy	Location;
	typedef typename Allocator::pointer		pointer;
	typedef std::vector<pointer>				Root;

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

	void		setopenbranch(int);
	void		addbranch();
	void		addbranch(pointer);
	void		removebranch();
	void		removebranch(int);
	void		replacebranch(pointer,int);
	void		clear();
	void		resize(int); 
	const Root&	root()const;
	
	Tree& operator= (const Tree&);  
	void swap( Tree<T,A>&);

	private:
	Allocator				_allocator; 
	Root					_root;
	int					_openbranch;
};

template<typename A,typename B, typename C, typename D>
bool operator==(const Tree<A,B>&,const Tree<A,B>&);


#undef __both__
#include"Tree.inl"
#endif




