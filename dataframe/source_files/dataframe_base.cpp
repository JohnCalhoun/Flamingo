//dataframe_base.cpp
#ifndef DATAFRAME_BASE_H
#define DATAFRAME_BASE_H

#include <location.cu> 
#include "traits.cpp"
#include "columns.cpp"
#include "iterator.cpp"
#include "addressbook.cpp"
#include <vector>
#include <array>
#include <iterator>
#include <tbb/queuing_rw_mutex.h>

class dataframeBase {
	public:
	typedef addressbook<dataframeBase>			AddressBook;
	typedef typename AddressBook::Key			Key;
	typedef typename AddressBook::Value		Value; 
	typedef typename AddressBook::iterator		iterator; 

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
	static AddressBook						_addressbook;
	Key									_key; 	
}; 
dataframeBase::AddressBook dataframeBase::_addressbook; 

#include"dataframe_base.inl"
#endif 

