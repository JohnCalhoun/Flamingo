//traits.cpp
#ifndef DATAFRAME_BOOK
#define DATAFRAME_BOOK

#include <map>
#include <boost/thread.hpp>
#include <mutex>
#include <threading/shared_mutex.cpp>

template<typename Object>
class addressbook {
	public:
	typedef unsigned int	Key;
	typedef Object*		Value;

	private:
	typedef std::map<Key,Value>		Map;

	typedef flamingo::threading::shared_mutex			Mutex;
	typedef flamingo::threading::shared_lock_guard<Mutex>	shared_guard;
	typedef flamingo::threading::lock_guard<Mutex>		lock_guard;

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

