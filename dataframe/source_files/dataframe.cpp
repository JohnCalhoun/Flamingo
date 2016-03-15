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
	typedef dataframeBase		base;

	public:
	typedef traits<Type...>						Traits; 
	
	typedef typename Traits::size_type			size_type;
	typedef dataframe_iterator<Type...>		iterator;
	typedef typename Traits::difference_type	difference_type;
	typedef typename Traits::reference			reference;
	typedef typename Traits::value_type		value_type;
	typedef typename Traits::pointer			pointer;
	typedef typename Traits::type_vector		type_vector;
	typedef typename column_tuple<Type...>::type		ColumnTuple;

	private:
	ColumnTuple		_column_tuple;

	public:
	dataframe();
	dataframe(const dataframe<Type...>&);
	dataframe(size_type,value_type);
	dataframe(size_type);
	dataframe(iterator,iterator);

	~dataframe(); 

	private:
	template<int n>
	typename column_tuple<Type...>::element<n>::type& column_access();

	ColumnTuple& tuple(); 
	const ColumnTuple& tuple_const()const;
	public:
	void assign(iterator,iterator);
	void assign(size_type,value_type);

	dataframe<Type...>& operator=(const dataframe<Type...>&);
	reference at(size_type);
	reference operator[](size_type);
	reference front();
	reference back();

	iterator begin();
	iterator end(); 

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

