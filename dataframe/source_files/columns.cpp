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
#include <tuple>
#include <HashedArrayTree.cu>
#include <traits.cpp>

namespace Flamingo {
namespace DataFrame {

template<typename T>
struct column  {
	typedef column_traits<T>				Traits; 
	typedef typename Traits::MemoryTuple	MemoryTuple;	
	typedef typename Traits::pointer		pointer;
	typedef pointer					iterator; 
	typedef typename Traits::const_pointer	const_pointer;
	typedef typename Traits::size_type		size_type; 
	typedef typename Traits::value_type	value_type;

	template<Memory::Region M>
		using Return=typename Traits::Return<M>; 

	template<Memory::Region M>
		using Memory2Type=memory2type<M>; 

	column();
	column(int);
	column(const column<T>& );
	column(pointer ,pointer );
	~column(); 

	template<typename Aloc>
	column(const thrust::device_vector<T,Aloc>&); 

	template<typename Aloc>
	column(const thrust::host_vector<T,Aloc>&); 

	template<Memory::Region M>
	void move();

	Memory::Region getlocation()const; 

	template<Memory::Region M>
	typename Return<M>::column& access(); 	

	pointer data(); 
	const_pointer data()const; 

	void swap(column<T>& );
	column<T>& operator=(const column<T>& );
	bool operator==(const column<T>&)const;
	bool operator!=(const column<T>&)const; 

	//--------vector functions
	
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

	template<typename iter>
	void copy_to_array(iter,pointer)const; 

	template<typename iter>
	void push_from_array(iter); 

	private:	
	Memory::Region	_location;
	MemoryTuple	_tuple; 

};

#include "columns.inl"

}
}
#endif 
