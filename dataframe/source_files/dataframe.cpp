//dataframe.cpp
#ifndef DATAFRAME
#define DATAFRAME

#include "columns.cpp"
#include "iterator.cpp"
#include "traits.cpp"
#include <vector>

template<class ... Type>
class dataframe : public traits<Type...> {
	//typedefs
	typedef std::vector<void*> branch;

	public:
	typedef dataframe_iterator<Type...>	iterator;

	private:
	branch _branch;

	public:
	dataframe();
	dataframe(dataframe);
	dataframe(size_type,value_type);
	dataframe(iterator,iterator);

	~dataframe(); 

	private:
	iterator row_access(INT n); 		

	template<int n,typename L>
	typename column_return<n,L,type_vector>::type column_access();

	template<int n,typename S,typename D>
	void move_column(); 	

	public:
	void assign(iterator,iterator);
	void assign(size_type,value_type);

	reference operator=(dataframe)
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
	
	bool operator==(const dataframe&,const dataframe&);
	bool operator!=(const dataframe&,const dataframe&);
};

#include"dataframe.inl"
#endif 

