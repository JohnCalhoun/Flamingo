//dataframe.cpp
#ifndef DATAFRAME
#define DATAFRAME

#include <location.cu> 
#include "traits.cpp"
#include "columns.cpp"
#include "iterator.cpp"
#include "addressbook.cpp"
#include <vector>
#include <array>

class dataframeBase {
	public:
	typedef addressbook<dataframeBase>			AddressBook;
	static AddressBook						_addressbook;
	static AddressBook& book(){return _addressbook;}; 	
}; 

template<class ... Type>
class dataframe : public dataframeBase{	

	//typedefs
	typedef dataframeBase					base;

	public:
	typedef typename traits<Type...>::size_type		size_type;
		
	typedef dataframe_iterator<Type...>			iterator;
	typedef typename traits<Type...>::difference_type	difference_type;
	typedef typename traits<Type...>::reference		reference;
	typedef typename traits<Type...>::value_type		value_type;
	typedef typename traits<Type...>::pointer		pointer;
	typedef typename traits<Type...>::type_vector	type_vector;
	typedef typename traits<Type...>::pointer_zip	zip_it;
	typedef typename traits<Type...>::ColumnArray	ColumnArray;

	private:
	ColumnArray		_column_array;

	public:
	dataframe();
	dataframe(const dataframe<Type...>&);
	dataframe(dataframe<Type...>&&);	
	dataframe(size_type,value_type);
	dataframe(size_type);
	dataframe(iterator,iterator);

	~dataframe(); 

	private:
	template<int n>
	typename column_return<n,Type...>::type* column_access();

	public:
	void assign(iterator,iterator);
	void assign(size_type,value_type);

	reference operator=(const dataframe<Type...>&);
	reference at(size_type)const;
	reference operator[](size_type);
	reference front()const;
	reference back()const;

	iterator begin()const;
	zip_it  begin_zip()const;
	iterator end()const; 
	zip_it end_zip()const;

	size_type size()const;
	size_type max_size()const;
	bool empty()const;
	void reserve(size_type);
	size_type capacity()const;

	void clear();
	iterator insert(iterator,value_type);
	iterator insert(iterator,iterator,iterator);

	iterator erase(iterator);
	iterator erase(iterator,iterator);

	void push_back(value_type);
	void pop_back();
	void resize(size_type);
	void resize(size_type,value_type);	
	void swap(dataframe<Type...>&); 
	
	bool operator==(const dataframe<Type...>&)const;
	bool operator!=(const dataframe<Type...>&)const;
};

#include"dataframe.inl"
#endif 

