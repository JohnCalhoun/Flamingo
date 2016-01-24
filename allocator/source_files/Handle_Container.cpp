// Handle_Container.cpp
#ifndef RESERVED_CONTAINER
#define RESERVED_CONTAINER

#include <mutex>
#include <vector>
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <functional>
#include <tr1/unordered_map>
#include <utility>
#include \
    "Handle.cpp"
#include <iostream>
/** \ingroup allocator-module
 */
template <typename T>
class Handle_Container {
    public:
     typedef Handle<T>* Handle_ptr;
     typedef typename std::tr1::unordered_map<int, Handle_ptr> Map;
     typedef typename Map::iterator iter;
     typedef typename Map::iterator iterator;
     typedef std::mutex Mutex;

    public:
     Map map;
     Mutex mut;
     ~Handle_Container();

     Handle_ptr find_handle(int);
     Handle_ptr find_and_remove_handle(int);
     void insert(Handle_ptr);
     Handle_ptr get_remove_any();
     std::vector<Handle_ptr> handle_list();

     bool empty();
};

template <typename T>
bool Handle_Container<T>::empty() {
     return map.empty();
}

template <typename T>
std::ostream& operator<<(std::ostream& out, Handle_Container<T>& c) {
     typedef typename Handle_Container<T>::Handle_ptr ptr;

     std::vector<ptr> vect = c.handle_list();
     std::for_each(vect.begin(), vect.end(), [&out](ptr h) mutable {
    int off = h->_offset;
    out << off << ' ';
     });
     return out;
};

template <typename T>
Handle_Container<T>::Handle_ptr Handle_Container<T>::get_remove_any() {
     std::lock_guard<Mutex> lock(mut);
     if (!map.empty()) {
          auto it = map.begin();
          Handle_ptr h = it->second;
          if (it != map.end()) {
               map.erase(it);
          } else {
               h = NULL;
          }
          return h;
     } else {
          return NULL;
     }
}

template <typename T>
Handle_Container<T>::~Handle_Container() {
     std::for_each(map.begin(), map.end(),
                   [](std::pair<int, Handle_ptr> p) { delete (p.second); });
}

template <typename T>
Handle_Container<T>::Handle_ptr Handle_Container<T>::find_handle(int offset) {
     std::lock_guard<Mutex> lock(mut);
     auto it = map.find(offset);
     if (it != map.end())
          return it->second;
     else
          return NULL;
}

template <typename T>
Handle_Container<T>::Handle_ptr Handle_Container<T>::find_and_remove_handle(
    int offset) {
     std::lock_guard<Mutex> lock(mut);
     auto it = map.find(offset);
     if (it != map.end()) {
          Handle_ptr h = it->second;
          map.erase(it);
          return h;
     } else {
          return NULL;
     }
}

template <typename T>
void Handle_Container<T>::insert(Handle_Container<T>::Handle_ptr h) {
     std::lock_guard<Mutex> lock(mut);
     int offset = h->_offset;
     map.insert(std::pair<int, Handle_ptr>(offset, h));
}

template <typename T>
std::vector<typename Handle_Container<T>::Handle_ptr>
Handle_Container<T>::handle_list() {
     typedef std::vector<Handle_ptr> vector;
     typedef typename vector::iterator iter_v;

     vector output(map.size());
     std::transform(map.begin(), map.end(), output.begin(),
                    [](std::pair<int, Handle_ptr> p) { return p.second; });
     std::sort(output.begin(), output.end());
     return output;
}

#endif
