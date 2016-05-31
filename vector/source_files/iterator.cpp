//HashedArrayTree.cpp
#ifndef HASHED_ARRAY_TREE_ITERATOR_CPP
#define HASHED_ARRAY_TREE_ITERATOR_CPP

#include<allocator.cu>
#include<cmath>
#include<type_traits>
#include "Tree.cu"
#include "cordinate.h"
#include "internal.h"
#include "reference.cu"
#define __both__ __host__ __device__ 

namespace Flamingo{
namespace Vector{

template<typename T,typename operation>
class Iterator : public operation {
	public:
	typedef std::random_access_iterator_tag		iterator_category;
	typedef Internal::Cordinate<T>			Cordinate; 
	typedef T*							pointer; 	
	typedef std::size_t						size_type;
	typedef std::ptrdiff_t					difference_type;
	typedef Iterator<T,operation>				iterator;
	typedef T								value_type; 

	typedef reference_wrapper<T>			reference;
	typedef reference_wrapper<T>			const_reference; 
	
	__both__ Iterator();
	__both__ Iterator(T* ptr):_cordinate(ptr){}; 

		    template<typename A>	
		    Iterator(const Tree<T,A>&); 

		    template<typename A>	
		    Iterator(Tree<T,A>&,Iterator&); 
	
	__both__ Iterator(const Iterator&);
	__both__ ~Iterator();

	__both__ void initalize(size_type);
	__both__ void initalize(size_type,size_type); 
	__both__ void initalize(size_type,size_type,pointer); 
	__both__ void initalize(Cordinate);
	
	__both__ Iterator<T,operation>& operator=(const Iterator<T,operation>&);
	__both__ bool operator==(const Iterator<T,operation>&) const;
	__both__ bool operator!=(const Iterator<T,operation>&) const;
	__both__ bool operator<(const Iterator<T,operation>&) const; 
	__both__ bool operator>(const Iterator<T,operation>&) const; 
	__both__ bool operator<=(const Iterator<T,operation>&) const; 
	__both__ bool operator>=(const Iterator<T,operation>&) const; 
	
	__both__ Iterator& operator++();
	__both__ Iterator operator++(int); 
	__both__ Iterator& operator--(); 
	__both__ Iterator operator--(int); 
	__both__ Iterator& operator+=(size_type); 
	
	__both__ Iterator<T,operation> operator+(size_type) const; 
	__both__ Iterator<T,operation>& operator-=(size_type);  
	__both__ Iterator<T,operation> operator-(size_type) const; 
	__both__ difference_type operator-(Iterator<T,operation>) const; 

	__both__ reference operator*();
	__both__ pointer operator->();
	__both__ reference operator[](size_type); //optional

		    template<typename U>
	__both__ operator U*(){return _cordinate.access(); } 
	public: 
	Cordinate			_cordinate;
};
#include "iterator.inl"

}
}

namespace std{
	
	template<typename T,typename op>
	struct remove_pointer<Flamingo::Vector::Iterator<T,op> >{
		typedef T type; 
	};
}

#undef __both__
#endif

















