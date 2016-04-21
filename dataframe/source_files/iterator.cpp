//iterator.cpp
#ifndef DATAFRAME_ITERATOR
#define DATAFRAME_ITERATOR

#include <iterator>
#include "columns.cpp"
#include "dataframe_traits.cpp"

namespace Flamingo{
namespace DataFrame{

template<typename ref_type,typename pointer_type, class ... Type>
class dataframe_iterator: public traits<Type...> {
	public:
	typedef typename traits<Type...>::size_type			size_type; 
	typedef typename traits<Type...>::difference_type		difference_type; 
	typedef ref_type								reference;
	typedef pointer_type							pointer;
	typedef typename traits<Type...>::range				range; 
	typedef typename column_tuple<Type...>::type			ColumnTuple;
	typedef std::random_access_iterator_tag				iterator_category;
	typedef dataframe_iterator<ref_type,pointer_type,Type...> self; 	


	pointer get_pointer()const;
	public:
	dataframe_iterator();
	dataframe_iterator(const self&);
	dataframe_iterator(ColumnTuple&); 
	dataframe_iterator(const  ColumnTuple&); 
	dataframe_iterator(pointer p):_pointer(p){}; 
	~dataframe_iterator();

	self& operator=(self);
	bool operator==(const self&) const;
	bool operator!=(const self&) const;
	bool operator<(const self&) const; 
	bool operator>(const self&) const; 
	bool operator<=(const self&) const; 
	bool operator>=(const self&) const; 
	
	self& operator++();
	self  operator++(int); 
	self& operator--(); 
	self  operator--(int);
 
	self& operator+=(size_type); 	
	self  operator+(size_type) const; 
	self& operator-=(size_type);  
	self  operator-(size_type) const; 
	difference_type operator-(const self&) const; 

	reference operator*();
	reference operator*()const;
	reference operator[](size_type); //optional

	void swap(self&); 

	template<int n>
	typename traits<Type...>::Return<n>::pointer_base get();

	explicit operator bool(); 

	private:
		pointer _pointer; 
};

#include "iterator.inl"

}//end dataframe
}//end flamingo
#endif 

