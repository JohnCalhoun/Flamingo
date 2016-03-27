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

class dataframeBase {
	public:
	typedef cordinator<dataframeBase>			Cordinator;
	typedef typename Cordinator::Key			Key;
	typedef typename Cordinator::Value			Value; 
	typedef typename Cordinator::iterator		iterator; 

	typedef tbb::queuing_rw_mutex				Mutex;
	typedef typename Mutex::scoped_lock		lock_guard; 

	dataframeBase();  	
	~dataframeBase(); 

	public:
	static Value find(Key);
	Key id();
	void id(int); 

	iterator begin();
	iterator end(); 

	lock_guard* use(Memory); 
	private:
	virtual void unsafe_move(Memory)=0;
	void request_move(Memory,Key);
	void force_move(Key); 

	std::tuple<	lock_guard*,
				bool> try_lock(bool); 
	lock_guard* lock(bool);
	void release(lock_guard*); 

	private: 
	static Cordinator						_cordinator;
	
	Key									_key; 	
	Mutex*								_mutex_ptr; 	
}; 
dataframeBase::Cordinator dataframeBase::_cordinator; 

#include"dataframe_base.inl"
#endif 

