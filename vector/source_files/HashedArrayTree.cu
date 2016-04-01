//HashedArrayTree.cpp
#ifndef HASHED_ARRAY_TREE_CPP
#define HASHED_ARRAY_TREE_CPP

#include<allocator.cu>
#include<mutex>
#include<cmath>
#include"Tree.cu"
#include<type_traits>
#include"internal.h"
#define __both__ __host__ __device__ 

template<typename T,typename L>
class HashedArrayTree {
	public:
	typedef typename allocation_policy<T,L>::allocator		allocator_type;
	typedef location<L>									Location;

	typedef typename allocator_type::value_type		value_type;
	typedef typename allocator_type::reference		reference;
	typedef typename allocator_type::const_reference	const_reference;
	typedef typename allocator_type::difference_type	difference_type;
	typedef typename allocator_type::size_type		size_type;
	typedef typename allocator_type::pointer		pointer;
	typedef Tree<int,allocator_type>				tree;

	typedef std::recursive_mutex		Mutex;
	
	//iterators
	typedef Internal::forward forward;
	typedef Internal::reverse reverse;
	typedef Internal::Cordinate<T,allocator_type> Cordinate;
	typedef Internal::UP UP;
	typedef Internal::DOWN DOWN;

	template<typename operation>
	class Iterator : public operation {
		public:
		typedef typename allocator_type::value_type		value_type;
		typedef typename allocator_type::reference		reference;
		typedef typename allocator_type::difference_type	difference_type;
		typedef typename allocator_type::size_type		size_type;
		typedef typename allocator_type::pointer		pointer;
		typedef std::random_access_iterator_tag			iterator_category;
		typedef typename tree::device_pointer			pointer_branch;

		Cordinate			_cordinate;
		tree*			_tree_ptr;	
		
		__both__ Iterator();
			    Iterator(tree&); 
		__both__ Iterator(const Iterator&);
		__both__ ~Iterator();
	
		__both__ void initalize(int);
		__both__ void initalize(int,int); 
		__both__ void initalize(int,int,pointer); 
		__both__ void initalize(Cordinate);
		__both__ void setWidth(int x);
		
		__both__ Iterator<operation>& operator=(const Iterator<operation>&);
		__both__ bool operator==(const Iterator<operation>&) const;
		__both__ bool operator!=(const Iterator<operation>&) const;
		__both__ bool operator<(const Iterator<operation>&) const; 
		__both__ bool operator>(const Iterator<operation>&) const; 
		__both__ bool operator<=(const Iterator<operation>&) const; 
		__both__ bool operator>=(const Iterator<operation>&) const; 
		
		__both__ Iterator& operator++();
		__both__ Iterator operator++(int); 
		__both__ Iterator& operator--(); 
		__both__ Iterator operator--(int); 
		__both__ Iterator& operator+=(size_type); 
		
		__both__ Iterator<operation> operator+(size_type) const; 
		__both__ Iterator<operation>& operator-=(size_type);  
		__both__ Iterator<operation> operator-(size_type) const; 
		__both__ difference_type operator-(Iterator<operation>) const; 

		__both__ reference operator*();
		__both__ pointer operator->();
		__both__ reference operator[](size_type); //optional
	};
	typedef Iterator<forward> const_iterator;
	typedef Iterator<forward> iterator;
	typedef Iterator<reverse> reverse_iterator;
	typedef Iterator<reverse> const_reverse_iterator;

	allocator_type		_allocator;
	Location			location; 
	Mutex			_mutex;
	tree				_tree;
	size_type			_size;	
	
	//construcors, destructors
	HashedArrayTree();
	HashedArrayTree(const HashedArrayTree&);
	HashedArrayTree(int,T);
	~HashedArrayTree();
	
	//internal functions
	void	resize(int); 
	void add_end(int); 
	void remove_end(int);
	template<typename D>
		void shift(iterator,int);
	template<typename I>
		void	copy(I,int,const T&); 
	template<typename I>
		void copy(I,int,I);
	
	int calculate_width(int);
	//interface	
	iterator begin();
	iterator end();
	const_iterator cbegin();
	const_iterator cend(); 
	reverse_iterator rbegin();
	reverse_iterator rend();
	const_reverse_iterator crbegin();
	const_reverse_iterator crend(); 
	
	HashedArrayTree& operator=(const HashedArrayTree&);
	template<typename A>
		HashedArrayTree& operator=(const HashedArrayTree<T,A>&);

	bool operator==(const HashedArrayTree&)const;
	bool operator!=(const HashedArrayTree&)const;

	void			swap(HashedArrayTree&);
	size_type		size()const;
	size_type		max_size()const;
	size_type		capacity()const;
	bool			empty()const; 
	
	void reserve(int);
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
//	template<class iter>
//		void		assign(iter, iter); 
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
	
	allocator_type get_allocator(); 
	//locking 
	void		lock();
	void		unlock();
	bool		try_lock();
};
//test between different arrays
//conversion between different arrays
#include"iterator.inl"
#include"HashedArrayTree.inl"

#undef __both__
#endif

















