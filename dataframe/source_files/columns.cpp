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
#include "columnbase.cpp"
#include <tuple>

template<Memory M>
struct memory2type{
	typedef boost::mpl::int_<0> type; 
};
template<>
struct memory2type<device>{
	typedef boost::mpl::int_<1> type; 
};
template<>
struct memory2type<pinned>{
	typedef boost::mpl::int_<2> type; 
};
template<>
struct memory2type<unified>{
	typedef boost::mpl::int_<3> type; 
};

template<typename T>
struct column  {
	typedef typename allocation_policy<T,device>::allocator device_allocator; 
	typedef typename allocation_policy<T,pinned>::allocator pinned_allocator; 
	typedef typename allocation_policy<T,unified>::allocator unified_allocator; 

	typedef thrust::device_vector<T,device_allocator>		device_column;
	typedef thrust::device_vector<T,pinned_allocator>		pinned_column;
	typedef thrust::device_vector<T,unified_allocator>	unified_column;
	typedef thrust::host_vector<T>					host_column;
	
	typedef boost::mpl::map<	
		boost::mpl::pair<typename memory2type<device>::type,device_column>,
		boost::mpl::pair<typename memory2type<host>::type,host_column>,
		boost::mpl::pair<typename memory2type<pinned>::type,pinned_column>,
		boost::mpl::pair<typename memory2type<unified>::type,unified_column>
	> map; 
	typedef std::tuple<	
					host_column,
					device_column,
					pinned_column,
					unified_column
				> MemoryTuple; 

	typedef T					value_type;
	typedef value_type*			pointer; 
	typedef const value_type*	const_pointer; 	

	template<Memory M>
	struct Return{
		typedef typename boost::mpl::at<map,typename memory2type<M>::type>::type column;
	};

	Memory		_location;
	MemoryTuple	_tuple; 

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
	typename Return<M>::column& access(); 	

	pointer data(); 
	const_pointer data()const; 

	void swap(column<T>& );
	column<T>& operator=(const column<T>& );
	bool operator==(const column<T>&)const;
	bool operator!=(const column<T>&)const; 
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
	typedef typename traits<Type...>::Return<n>::type		base; 
	typedef typename column<base>::type				type;  
};

template<typename T>
struct type2column {
	typedef column<T> type; 
};

template<class ... Type>
struct column_tuple {
	typedef traits<Type...> Traits; 

	typedef typename Traits::type_vector vec; 
	typedef typename transform<vec,type2column<_1> >::type col_vec;
	typedef typename vec2tuple<Traits::_numCol-1,col_vec>::type type; 

	template<int n>
	struct element {
		typedef boost::mpl::int_<n> position;
		typedef typename boost::mpl::at<col_vec,position>::type type; 
	};
};

#include "columns.inl"
#endif 
