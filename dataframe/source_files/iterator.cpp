//iterator.cpp
#ifndef DATAFRAME_ITERATOR
#define DATAFRAME_ITERATOR

#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include "traits.cpp"
#include <iterator>
#include <boost/mpl/for_each.hpp>


template<class ... Type>
class dataframe_iterator: public traits<Type...> {
	public:
	typedef typename traits<Type...>::size_type			size_type; 
	typedef typename traits<Type...>::difference_type		difference_type; 
	typedef typename traits<Type...>::reference			reference;
	typedef typename traits<Type...>::pointer			pointer;
	typedef typename traits<Type...>::range				range; 
 
	typedef std::random_access_iterator_tag			iterator_category;
	
	pointer _pointer; 
	pointer get_pointer()const;

	public:
	dataframe_iterator();
	dataframe_iterator(const dataframe_iterator<Type...>&);
	dataframe_iterator(pointer p):_pointer(p){}; 
	~dataframe_iterator();

	dataframe_iterator<Type...>& operator=(const dataframe_iterator<Type...>&);
	bool operator==(const dataframe_iterator<Type...>&) const;
	bool operator!=(const dataframe_iterator<Type...>&) const;
	bool operator<(const dataframe_iterator<Type...>&) const; 
	bool operator>(const dataframe_iterator<Type...>&) const; 
	bool operator<=(const dataframe_iterator<Type...>&) const; 
	bool operator>=(const dataframe_iterator<Type...>&) const; 
	
	dataframe_iterator<Type...>& operator++();
	dataframe_iterator<Type...>  operator++(int); 
	dataframe_iterator<Type...>& operator--(); 
	dataframe_iterator<Type...>  operator--(int);
 
	dataframe_iterator<Type...>& operator+=(size_type); 	
	dataframe_iterator<Type...>  operator+(size_type) const; 
	dataframe_iterator<Type...>& operator-=(size_type);  
	dataframe_iterator<Type...>  operator-(size_type) const; 
	difference_type operator-(const dataframe_iterator<Type...>&) const; 

	reference operator*();
	reference operator[](size_type); //optional

	template<int n>
	typename traits<Type...>::Return<n>::pointer_base get();
};

#include "iterator.inl"

#endif 

