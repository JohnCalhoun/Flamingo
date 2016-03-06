//columns.cpp
#ifndef COLUMNS
#define COLUMNS
#include <allocator.cu>
#include <location.cu> 
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <boost/mpl/map.hpp> 
#include <boost/mpl/pair.hpp>
#include <type_traits>
#include <boost/mpl/int.hpp>

template<Memory M>
struct memory2type{
	typedef boost::mpl::int_<1> type; 
};
template<>
struct memory2type<device>{
	typedef boost::mpl::int_<2> type; 
};
template<>
struct memory2type<pinned>{
	typedef boost::mpl::int_<3> type; 
};
template<>
struct memory2type<unified>{
	typedef boost::mpl::int_<4> type; 
};

template<typename T>
struct column : public columnbase {
	typedef thrust::device_vector<T,typename allocation_policy<T,device>::allocator>	device_column;
	typedef thrust::device_vector<T,typename allocation_policy<T,pinned>::allocator>	pinned_column;
	typedef thrust::device_vector<T,typename allocation_policy<T,unified>::allocator>	unified_column;
	typedef thrust::host_vector<T>		host_column;
	typedef T				value_type;
	typedef value_type*		pointer; 
	
	typedef boost::mpl::map<	
					boost::mpl::pair<typename memory2type<device>::type,device_column>,
					boost::mpl::pair<typename memory2type<host>::type,host_column>,
					boost::mpl::pair<typename memory2type<pinned>::type,pinned_column>,
					boost::mpl::pair<typename memory2type<unified>::type,unified_column>
				> map; 
	template<Memory M>
	struct Return{
		typedef typename boost::mpl::at<map,typename memory2type<M>::type>::type raw;
		typedef raw* type; 
	};
	

	Memory _location;
	void* _ptr; 

	column();
	column(int);
	column(const column<T>& );
	column(pointer ,pointer);
	~column(); 

	template<typename Aloc>
	column(const thrust::device_vector<T,Aloc>&); 

	template<typename Aloc>
	column(const thrust::host_vector<T,Aloc>&); 

	template<Memory M>
	void move();

	Memory getlocation()const; 

	template<Memory M>
	typename Return<M>::type access(); 	
	
	void* access_raw(); 

	void swap(column<T>& );
	column<T>& operator=(const column<T>& );
	//--------vector functions
	typedef typename host_column::size_type size_type; 
	
	size_type size()const;
	size_type max_size()const;
	bool empty()const;
	void reserve(size_type);
	size_type capacity()const;

	void fill(T);
	template<typename iter>
	void copy(iter,iter); 
	void clear(); 

	void assign(size_type, value_type);

	template<typename iter>
	void assign(iter,iter);

	template<typename iter>
	iter insert(iter,value_type);

	template<typename iter_pos,typename iter>
	void insert(iter_pos,iter,iter);

	template<typename iter>
	iter erase(iter); 

	template<typename iter>
	iter erase(iter,iter); 

	void resize(size_type); 
	void resize(size_type,value_type);
};

template<int n,class ... Type>
struct column_return{
	typedef typename traits<Type...>::Return<n>::type base; 

	typedef typename column<base>::type type;  
};








#include "columns.inl"
#endif 
