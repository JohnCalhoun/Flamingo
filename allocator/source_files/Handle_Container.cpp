// Handle_Container.cpp
#ifndef RESERVED_CONTAINER
#define RESERVED_CONTAINER

#include <vector>
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <functional>
#include <tbb/concurrent_unordered_map.h>
#include <utility>
#include "Handle.cpp"
#include <iostream>
#include <tbb/queuing_rw_mutex.h>

namespace Flamingo{
namespace Memory{

template <typename T>
class Handle_Container {
     public:
     typedef Handle<T>*				Handle_ptr;

     typedef tbb::concurrent_unordered_map<int, Handle_ptr> Map;
	typedef typename Map::value_type	value_type;
     typedef typename Map::iterator	iterator;

	typedef tbb::queuing_rw_mutex			Mutex; 
	typedef typename Mutex::scoped_lock	scoped_lock; 

     public:
     ~Handle_Container();

     Handle_ptr	find_handle(int);
     Handle_ptr	find_and_remove_handle(int);
     void			insert(Handle_ptr);
     Handle_ptr	get_remove_any();

     std::vector<Handle_ptr>	handle_list();
     bool					empty();

	private:
     Map		map;
	Mutex	_mutex; 
};

#include "Handle_Container.inl"
}//end Memory
}//end flamingo

#endif
