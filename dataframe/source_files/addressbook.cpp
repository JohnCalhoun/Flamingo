//addressbook.cpp
#ifndef DATAFRAME_BOOK
#define DATAFRAME_BOOK

#include <unordered_map>
#include <boost/thread.hpp>
#include <mutex>
#include <tbb/concurrent_hash_map.h>
#include "traits.cpp"

template<typename Object>
class addressbook {
	public:
	typedef typename traits<int>::size_type			Key;
	typedef Object*							Value;

	private:
	typedef tbb::concurrent_hash_map<Key,Value>		Map;
	typedef typename Map::const_accessor			r_accessor; 
	typedef typename Map::accessor				w_accessor; 
	typedef typename Map::value_type				value_type; 

	public:	
	typedef typename Map::iterator				iterator; 
	
	private:
	//data members
	Map		_map;

	public:
	//constructors
	addressbook();
	~addressbook();

	//member functions
	private:
	Key objectToKey(Object*); 

	public:
	Key insert(Object*);
	Key insert(Key,Value);  
	void remove(Key); 
	Value find(Key);
	void change(Key,Key);
	
	iterator begin();
	iterator end(); 
};

#include "addressbook.inl"
#endif 

