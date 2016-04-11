// free_list.hpp
#ifndef FREE_LIST
#define FREE_LIST

#include <atomic>
#include <mutex>
#include <list>
#include <vector>
#include <deque>
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <iostream>
#include <functional>
#include <boost/unordered_map.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>

#include "Handle.cpp"
#include "Free_Container.cpp"
#include "Handle_Container.cpp"

namespace Flamingo{
namespace Memory {

template <typename T>
class Free_List {
    public:
     typedef std::size_t size_type;
     typedef Handle<T>* pointer;

     bool vacant();
	bool allFree();

     void		split(pointer&, size_type);
     pointer	find_free_handle(size_type /*size*/);
     void		add_reserved_handle(pointer);

     pointer	find_reserved_handle(int /*offset*/);
     void		combine(pointer&);
     void		add_free_handle(pointer);

	private:
     Handle_Container<T>		reserved_container;
     Free_Container<T>		free_container;
};

#include "Free_List.inl"
}//end Memory
}//end Flamingo
#endif
