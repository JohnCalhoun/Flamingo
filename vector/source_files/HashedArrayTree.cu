//HashedArrayTree.cpp
#ifndef HASHED_ARRAY_TREE_CPP
#define HASHED_ARRAY_TREE_CPP

#include<allocator.cu>
#include<cmath>
#include"Tree.cu"
#include<type_traits>
#include "cordinate.h"
#include"internal.h"
#include "reference.cu"
#include "iterator.cpp"
#define __both__ __host__ __device__ 

namespace Flamingo {
namespace Vector {

template<typename T>
struct HashedArrayTree_base {	
	typedef Vector::reference_wrapper<T>			reference;
	typedef Vector::reference_wrapper<T>		const_reference;

	typedef Internal::Cordinate<T> Cordinate;

	typedef Internal::forward forward;
	typedef Internal::reverse reverse;

	typedef Iterator<T,forward> const_iterator;
	typedef Iterator<T,forward> iterator;
	typedef Iterator<T,reverse> reverse_iterator;
	typedef Iterator<T,reverse> const_reverse_iterator;
};
	
template<typename T,Memory::Region M>
class HashedArrayTree : public HashedArrayTree_base<T>{
	using Base=HashedArrayTree_base<T>; 

	public:
	typedef typename Base::reference		reference;
	typedef typename Base::const_reference	const_reference;
	typedef typename Base::Cordinate		Cordinate;

	private:
	typedef Internal::UP UP;
	typedef Internal::DOWN DOWN;

	public:
	typedef typename Base::const_iterator			const_iterator;
	typedef typename Base::iterator				iterator;
	typedef typename Base::reverse_iterator			reverse_iterator;
	typedef typename Base::const_reverse_iterator	const_reverse_iterator;

	public:
	typedef typename 
		Memory::allocation_policy<T,M>::allocator	allocator_type;
	typedef Memory::location<M>					Location;

	typedef typename allocator_type::value_type		value_type;
	typedef typename allocator_type::difference_type	difference_type;
	typedef typename allocator_type::size_type		size_type;
	typedef typename allocator_type::pointer		pointer;
	typedef Tree<T,allocator_type>				tree;
	
	//iterators
	allocator_type		_allocator;
	Location			location; 
	tree				_tree;
	size_type			_cap;	
	size_type			_count;
	
	//construcors, destructors
	HashedArrayTree();
	HashedArrayTree(const HashedArrayTree&);
	HashedArrayTree(size_type);
	HashedArrayTree(size_type,T);
	~HashedArrayTree();
	
	//internal functions
	void	resize(size_type); 
	iterator add_end(size_type,iterator); 
	void remove_end(size_type);
	template<typename D>
		void shift(iterator,size_type);
	template<typename I>
		void	copy(I,size_type,const T&); 
	template<typename I>
		void copy(I,size_type,I);
	
	size_type calculate_width(size_type);
	//interface	
	iterator begin();
	iterator end();
	const_iterator cbegin()const;
	const_iterator begin()const{return cbegin();};
	const_iterator cend()const; 
	reverse_iterator rbegin();
	reverse_iterator rend();
	const_reverse_iterator crbegin()const;
	const_reverse_iterator crend()const; 
	
	template<Memory::Region O>
		HashedArrayTree& operator=(const HashedArrayTree<T,O>&);

	bool operator==(const HashedArrayTree&)const;
	bool operator!=(const HashedArrayTree&)const;

	void			swap(HashedArrayTree&);
	size_type		size()const;
	size_type		max_size()const;
	size_type		capacity()const;
	bool			empty()const; 
	
	void reserve(size_type);
	void shrink_to_fit(); 

	//template<class ...Args>
	//	iterator	emplace(const_iterator, Args...); 
     iterator		insert(const_iterator, const T&); 
	iterator		insert(const_iterator, T&&); 
	iterator		insert(const_iterator, size_type, T&); 
	
	template<class iter>
		iterator	insert(const_iterator, iter, iter); 
	iterator		insert(const_iterator, std::initializer_list<T>); 
	iterator		erase(const_iterator); 
	iterator		erase(const_iterator, const_iterator); 
	void			clear(); 
	template<class iter>
		void		assign(iter, iter); 
	void			assign(std::initializer_list<T>); 
	void			assign(size_type, const T&); 
	

	void push_back(const T&);
	void push_back(T&&);
	void pop_back();
//	template<class... Args>
//		void emplace_back(Args&&.. args); 

	reference operator[](size_type); 	
	reference at(size_type);
	reference front(); 
	reference back();

	const_reference operator[](size_type)const; 	
	const_reference at(size_type)const;
	const_reference front()const; 
	const_reference back()const;

	allocator_type get_allocator(); 
};
//test between different arrays
//conversion between different arrays
#include "HashedArrayTree.inl"

}
}
#undef __both__
#endif

















