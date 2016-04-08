// Free_Container.cpp
#ifndef FREE_CONTAINER
#define FREE_CONTAINER

#include <mutex>
#include <vector>
#include <deque>
#include <algorithm>
#include <iterator>
#include <functional>
#include <mutex>
#include \
    "Handle.cpp"
#include \
    "Handle_Container.cpp"

#include <iostream>
/** \ingroup allocator-module
 */
template <typename T>
class Free_Container {
    public:
     typedef Handle_Container<T> Row;
     typedef std::deque<Row*> Column;

     typedef std::mutex Lock;
     typedef std::lock_guard<Lock> lock_guard;

     typedef std::size_t size_type;
     typedef Handle<T>* Handle_ptr;
     typedef std::vector<Handle_ptr> vector;

     Lock lock;
     Column column;
     int max_order;

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
};
template <typename T>
bool Free_Container<T>::empty() {
     std::vector<bool> bool_vector(column.size());
     std::transform(column.begin(), column.end(), bool_vector.begin(),
                    [](Row* r_ptr) { return r_ptr->empty(); });

     auto it = std::find(bool_vector.begin(), bool_vector.end(), false);

     return it == bool_vector.end();
}

template <typename T>
Free_Container<T>::Free_Container() {
     max_order = 0;
     add_order();
};

template <typename T>
Free_Container<T>::~Free_Container() {
     std::for_each(column.begin(), column.end(), [](Row* r) { delete r; });
};

template <typename T>
std::ostream& operator<<(std::ostream& out, Free_Container<T>& container) {
     typedef typename Free_Container<T>::Row* row;
     std::for_each((container.column).begin(), (container.column).end(),
                   [&out](row r) mutable { out << '|' << *r; });
     out << '|';
     return out;
};

template <typename T>
void Free_Container<T>::add_order() {
     column.push_back(new Row);
     max_order++;
}

template <typename T>
int Free_Container<T>::size2order(std::size_t size) {
     auto log = std::log2(size);
     return std::ceil(log);
}

template <typename T>
Free_Container<T>::Handle_ptr Free_Container<T>::find_remove_handle(
    std::size_t size) {
     int max_index = size2order(size);
     if (max_index <= max_order) {
          Handle_ptr h;
		if(max_index>0){
			for (int i = max_index; i <= max_order; i++) {
				h = column[i - 1]->get_remove_any();
				if (h)
					break;
			}
		}else{
			h=column[0]->get_remove_any(); 
		}
          return h;
     } else {
          return NULL;
     }
}

template <typename T>
bool Free_Container<T>::find_remove_handle(std::size_t size, int offset) {
     int index = size2order(size);
     Handle_ptr h_ptr = column[index - 1]->find_and_remove_handle(offset);
     return bool(h_ptr);
}

template <typename T>
Free_Container<T>::Handle_ptr Free_Container<T>::find_return_handle(
    std::size_t size, int offset) {
     int index = size2order(size);
     if (!column.empty()) {
          if (index <= max_order) {
               Handle_ptr h_ptr = column[index - 1]->find_and_remove_handle(offset);
               return h_ptr;
          } else {
               return NULL;
          }
     } else {
          return NULL;
     }
}

template <typename T>
void Free_Container<T>::insert(Free_Container<T>::Handle_ptr h) {
     int size = h->_size;
     int index = size2order(size);
     while (index > max_order) {
          lock_guard guard(lock);
          if (index > max_order)
               add_order();
     }

     column[index - 1]->insert(h);
}

template <typename T>
Free_Container<T>::vector Free_Container<T>::handle_list() {
     typedef typename vector::iterator iter_v;
     vector output;
     std::for_each(column.begin(), column.end(), [&, output](Row* r) mutable {
    vector vect = r->handle_list();
    std::copy(vect.begin(), vect.end(), std::back_inserter(output));
     });
     std::sort(output.begin(), output.end());
     return output;
}

#endif
