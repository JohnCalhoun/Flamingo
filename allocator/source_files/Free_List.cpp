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

#include \
    "Handle.cpp"
#include \
    "Free_Container.cpp"
#include \
    "Handle_Container.cpp"
/** \ingroup allocator-module
 */
template <typename T>
class Free_List {
    public:
     typedef std::size_t size_type;
     typedef Handle<T>* pointer;

     Handle_Container<T> reserved_container;
     Free_Container<T> free_container;

     bool vacant();

     void split(pointer&, size_type);
     pointer find_free_handle(size_type /*size*/);
     void add_reserved_handle(pointer);

     pointer find_reserved_handle(int /*offset*/);
     void combine(pointer&);
     void add_free_handle(pointer);
};

template <typename T>
std::ostream& operator<<(std::ostream& out, Free_List<T>& free_list) {
     std::cout << "Reserved Container" << '\n';
     std::cout << '|' << free_list.reserved_container << '|' << '\n';
     std::cout << "Free Container" << '\n';
     std::cout << free_list.free_container << '\n';
     return out;
}

template <typename T>
bool Free_List<T>::vacant() {
     return !(free_container.empty());
}

template <typename T>
void Free_List<T>::add_free_handle(Free_List<T>::pointer p) {
     free_container.insert(p);
}
template <typename T>
void Free_List<T>::add_reserved_handle(Free_List<T>::pointer p) {
     reserved_container.insert(p);
}

template <typename T>
Free_List<T>::pointer Free_List<T>::find_reserved_handle(int offset) {
     return reserved_container.find_and_remove_handle(offset);
};

template <typename T>
void Free_List<T>::combine(Free_List<T>::pointer& h) {
     int buddy_offset = h->buddy_offset();
     int size = h->_size;
     pointer buddy = free_container.find_return_handle(size, buddy_offset);
     if (buddy) {
          h->combine(*buddy);
          delete buddy;
          combine(h);
     }
};

template <typename T>
void Free_List<T>::split(Free_List<T>::pointer& p,
                         Free_List<T>::size_type size) {
     if (p->_size / 2 >= size) {
          p->_size /= 2;
          pointer buddy = new Handle<T>(*p);
          buddy->_offset = p->buddy_offset();
          free_container.insert(buddy);
          split(p, size);
     }
}

template <typename T>
Free_List<T>::pointer Free_List<T>::find_free_handle(
    Free_List<T>::size_type size) {
     return free_container.find_remove_handle(size);
}
#endif
