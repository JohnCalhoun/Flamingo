//iterator.cpp
#ifndef DATAFRAME_ITERATOR
#define DATAFRAME_ITERATOR

#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include "traits.cpp"
#include <iterator>

template<class ... Type>
class dataframe_iterator: public traits<Type...> {
	public:
	typedef typename traits<Type...>::size_type size_type; 
	typedef typename traits<Type...>::difference_type difference_type; 
	typedef typename traits<Type...>::reference reference;

 
	typedef std::random_access_iterator_tag			iterator_category;
	
	 dataframe_iterator();
	 dataframe_iterator(const dataframe_iterator<Type...>&);
	 ~dataframe_iterator();

	 dataframe_iterator<Type...>& operator=(const dataframe_iterator<Type...>&);
	 bool operator==(const dataframe_iterator<Type...>&) const;
	 bool operator!=(const dataframe_iterator<Type...>&) const;
	 bool operator<(const dataframe_iterator<Type...>&) const; 
	 bool operator>(const dataframe_iterator<Type...>&) const; 
	 bool operator<=(const dataframe_iterator<Type...>&) const; 
	 bool operator>=(const dataframe_iterator<Type...>&) const; 
	
	 dataframe_iterator& operator++();
	 dataframe_iterator operator++(int); 
	 dataframe_iterator& operator--(); 
	 dataframe_iterator operator--(int); 
	 dataframe_iterator& operator+=(size_type); 
	
	 dataframe_iterator<Type...> operator+(size_type) const; 
	 dataframe_iterator<Type...>& operator-=(size_type);  
	 dataframe_iterator<Type...> operator-(size_type) const; 
	 difference_type operator-(dataframe_iterator) const; 

	 reference operator*();
	 reference operator[](size_type); //optional
};











#endif 

