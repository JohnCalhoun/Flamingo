// Free_Container.cpp
#ifndef FREE_CONTAINER
#define FREE_CONTAINER

#include <vector>
#include <deque>
#include <algorithm>
#include <iterator>
#include <functional>
#include <mutex>
#include "Handle.cpp"
#include "Handle_Container.cpp"
#include <iostream>
#include <tbb/concurrent_vector.h>
namespace Flamingo {
namespace Memory {

template <typename T>
class Free_Container {
    public:
     typedef Handle_Container<T> Row;
     typedef tbb::concurrent_vector<Row*> Column;

     typedef std::size_t size_type;
     typedef Handle<T>* Handle_ptr;
     typedef std::vector<Handle_ptr> vector;

    public:
     ~Free_Container();
     Free_Container();

     void add_order();
     int size2order(std::size_t);

     Handle_ptr find_remove_handle(std::size_t);
     bool find_remove_handle(std::size_t, int);
     Handle_ptr find_return_handle(std::size_t, int);

     void insert(Handle_ptr);

     std::vector<Handle_ptr> handle_list();
     bool empty();

	private:
     Column	column;
     int		max_order;
};

#include "Free_Container.inl"
}//end memory
}//end Flamingo
#endif
