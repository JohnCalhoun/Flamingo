//dataframe.cpp
#ifndef DATAFRAME
#define DATAFRAME

#include <location.cu> 
#include "columns.cpp"
#include "iterator.cpp"
#include "traits.cpp"
#include <vector>
#include <array>

template<class ... Type>
class dataframe {	
	//typedefs
	typedef std::vector<void*> branch;
	typedef std::array<Memory,sizeof...(Type)> locations; 

	public:
	typedef dataframe_iterator<Type...>	iterator;

	typedef typename traits<Type...>::size_type		size_type;
	typedef typename traits<Type...>::difference_type	difference_type;
	typedef typename traits<Type...>::reference		reference;
	typedef typename traits<Type...>::value_type		value_type;
	typedef typename traits<Type...>::type_vector	type_vector;

	private:
	branch _branch;

	public:
	dataframe();
	dataframe(const dataframe&);
	dataframe(size_type,value_type);
	dataframe(iterator,iterator);

	~dataframe(); 

	private:
	iterator row_access(size_type n); 		

	template<int n,Memory M>
	typename traits<Type...>::column_return<n,M>::type column_access();

	template<int n,typename S,typename D>
	void move_column(); 	

	public:
	void assign(iterator,iterator);
	void assign(size_type,value_type);

	reference operator=(dataframe);
	reference at();
	reference operator[](size_type);
	reference front();
	reference back();

	iterator being();
	iterator end(); 

	size_type size();
	size_type max_size();
	bool empty();
	void reserve();
	size_type capacity();

	void clear();
	iterator insert(iterator,value_type);
	iterator insert(iterator,iterator,iterator);

	iterator erase(iterator);
	iterator erase(iterator,iterator);

	void push_back(value_type);
	void push_front(value_type);
	void pop_back();
	void pop_front();
	void resize(size_type);
	void resize(size_type,value_type);	
	void swap(dataframe&); 
	
	bool operator==(const dataframe<Type...>&);
	bool operator!=(const dataframe<Type...>&);
};

#include"dataframe.inl"
#endif 

