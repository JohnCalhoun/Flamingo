//dataframe_base.cpp
#ifndef DATAFRAME_BASE_H
#define DATAFRAME_BASE_H

#include <location.cu> 
#include "traits.cpp"
#include "columns.cpp"
#include "iterator.cpp"
#include "cordinator.cpp"
#include <vector>
#include <array>
#include <iterator>
#include <tbb/queuing_rw_mutex.h>

namespace Flamingo {
namespace DataFrame {

class dataframeBase {
	private:
	typedef tbb::queuing_rw_mutex				Mutex;
	typedef typename Mutex::scoped_lock		scoped_lock;
	public:
	typedef std::shared_ptr<scoped_lock>		lock_guard; 
	typedef cordinator<dataframeBase,lock_guard>			Cordinator;

	public:
	typedef typename Cordinator::Key			Key;
	typedef typename Cordinator::Value			Value; 

	dataframeBase();  	
	dataframeBase(const dataframeBase&); 
	~dataframeBase(); 

	public:
	static Value find(Key);
	Key id();
	void id(int); //value must be unique  

	lock_guard use(Memory::Region);
	virtual Memory::Region location()const=0; 	
	virtual size_t device_size()const=0; 
	void release(lock_guard&); 
	dataframeBase& operator=(const dataframeBase& other){return *this;}; 
	virtual void unsafe_move(Memory::Region)=0;
	std::tuple<	lock_guard,
				bool> try_lock(bool); 

	private:
	lock_guard lock(bool);

	private: 
	static Cordinator						_cordinator;
	
	Key									_key; 	
	Mutex*								_mutex_ptr; 	
}; 
dataframeBase::Cordinator dataframeBase::_cordinator; 

#include"dataframe_base.inl"
}//end dataframe
}//end flamingo 
#endif 

