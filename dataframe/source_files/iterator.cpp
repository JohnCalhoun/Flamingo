//iterator.cpp
#ifndef DATAFRAME_ITERATOR
#define DATAFRAME_ITERATOR

#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include "traits.cpp"

template<class ... Type>
class dataframe_iterator : public traits<Type...> {
	public:

	typedef typename allocator_type::pointer		pointer;
	typedef std::random_access_iterator_tag			iterator_category;
	typedef typename tree::device_pointer			pointer_branch;

	Cordinate			_cordinate;
	tree*			_tree_ptr;	
	
	 dataframe_iterator();
	 dataframe_iterator(const dataframe_iterator&);
	 ~dataframe_iterator();

	 void initalize(int);
	 void initalize(int,int); 
	 void initalize(int,int,pointer); 
	 void initalize(Cordinate);
	 void setWidth(int x);
	
	 dataframe_iterator<operation>& operator=(const dataframe_iterator<operation>&);
	 bool operator==(const dataframe_iterator<operation>&) const;
	 bool operator!=(const dataframe_iterator<operation>&) const;
	 bool operator<(const dataframe_iterator<operation>&) const; 
	 bool operator>(const dataframe_iterator<operation>&) const; 
	 bool operator<=(const dataframe_iterator<operation>&) const; 
	 bool operator>=(const dataframe_iterator<operation>&) const; 
	
	 dataframe_iterator& operator++();
	 dataframe_iterator operator++(int); 
	 dataframe_iterator& operator--(); 
	 dataframe_iterator operator--(int); 
	 dataframe_iterator& operator+=(size_type); 
	
	 dataframe_iterator<operation> operator+(size_type) const; 
	 dataframe_iterator<operation>& operator-=(size_type);  
	 dataframe_iterator<operation> operator-(size_type) const; 
	 difference_type operator-(dataframe_iterator<operation>) const; 

	 reference operator*();
	 pointer operator->();
	 reference operator[](size_type); //optional
};











#endif 

