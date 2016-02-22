//traits.cpp
#ifndef DATAFRAME_BOOK
#define DATAFRAME_BOOK

#include <map>
#include <boost/thread.hpp>
#include <mutex>

template<typename T>
class shared_lock_guard{	
	T* _lock; 

	shared_lock_guard(T&);
	~shared_lock_guard();
};

template<typename Object>
class addressbook {
	public:
	typedef unsigned int	Key;
	typedef Object*		Value;

	private:
	typedef std::map<Key,Value>		Map;
	typedef boost::shared_mutex		Mutex;
	typedef shared_lock_guard<Mutex>	shared_guard;
	typedef std::lock_guard<Mutex>	lock_guard;

	//data members
	Map		_map;
	Mutex	_mutex;

	public:
	//constructors
	addressbook();
	~addressbook();

	//member functions
	private:
	Key objectToKey(const Object&); 

	public:
	template<typename T>
	void insert(const T&); 

	void remove(Key); 

	template<typename T>
	T* find(Key); 
};

#include "addressbook.inl"
#endif 

