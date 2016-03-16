//addressbook.cpp
#ifndef DATAFRAME_BOOK
#define DATAFRAME_BOOK

#include <unordered_map>
#include <boost/thread.hpp>
#include <mutex>
#include <threading/shared_mutex.cpp>
#include "traits.cpp"

template<typename Object>
class addressbook {
	public:
	typedef typename traits<int>::size_type		Key;
	typedef Object*						Value;
	typedef std::unordered_map<Key,Value>		Map;
	typedef typename Map::iterator	iterator; 

	private:

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
	Key objectToKey(Object*); 

	public:
	Key insert(Object*);
	void insert(Key,Value);  
	void remove(Key); 
	Value find(Key);
	void change(Key,Key); 
};

#include "addressbook.inl"
#endif 

