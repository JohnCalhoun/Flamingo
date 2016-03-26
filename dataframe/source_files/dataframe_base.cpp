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

	dataframeBase();  	
	~dataframeBase(); 

	static Value find(Key);
	Key id();
	void id(int); 

	iterator begin();
	iterator end(); 

	virtual void move(Memory)=0;
	void request_move(Memory,Key);
	void force_move(Key); 
	private: 
	static Cordinator						_cordinator;
	Key									_key; 	
}; 
dataframeBase::Cordinator dataframeBase::_cordinator; 

#include"dataframe_base.inl"
#endif 

