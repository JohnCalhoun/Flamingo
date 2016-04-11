// Handle_Container.inl
template <typename T>
bool Handle_Container<T>::empty() {
	bool write=false;
	scoped_lock lock(_mutex,write); 

     return map.empty();
}

template <typename T>
std::ostream& operator<<(std::ostream& out, Handle_Container<T>& c) {

     typedef typename Handle_Container<T>::Handle_ptr ptr;
     std::vector<ptr> vect = c.handle_list();
     std::for_each(vect.begin(), vect.end(), 
			[&out](ptr h) mutable {
			    int off = h->_offset;
			    out << off << ' ';
     });
     return out;
};

template <typename T>
Handle_Container<T>::Handle_ptr Handle_Container<T>::get_remove_any() {
	bool write=true;
	scoped_lock lock(_mutex,write); 

     if (!map.empty()) {
          auto it = map.begin();
          Handle_ptr h = it->second;
          if (it != map.end()) {
               map.unsafe_erase(it);
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
	bool write=true;
	scoped_lock lock(_mutex,write); 

     std::for_each(map.begin(), map.end(),
                   [](value_type p) { delete (p.second); });
}

template <typename T>
Handle_Container<T>::Handle_ptr Handle_Container<T>::find_handle(int offset) {
	bool write=false;
	scoped_lock lock(_mutex,write); 

     auto it = map.find(offset);
     if (it != map.end())
          return it->second;
     else
          return NULL;
}

template <typename T>
Handle_Container<T>::Handle_ptr 
	Handle_Container<T>::find_and_remove_handle(int offset) 
{
	bool write=true;
	scoped_lock lock(_mutex,write); 

     auto it = map.find(offset);
     if (it != map.end()) {
          Handle_ptr h = it->second;
		map.unsafe_erase(it);
          return h;
     } else {
          return NULL;
     }
}

template <typename T>
void Handle_Container<T>::insert(Handle_Container<T>::Handle_ptr h) {
	bool write=false;
	scoped_lock lock(_mutex,write); 

     int offset = h->_offset;
     map.insert(std::pair<int, Handle_ptr>(offset, h));
}

template <typename T>
std::vector<typename Handle_Container<T>::Handle_ptr>
Handle_Container<T>::handle_list() {
	bool write=false;
	scoped_lock lock(_mutex,write); 

     typedef std::vector<Handle_ptr> vector;
     typedef typename vector::iterator iter_v;

     vector output(map.size());
     std::transform(map.begin(), map.end(), output.begin(),
                    [](std::pair<int, Handle_ptr> p) { return p.second; });
     std::sort(output.begin(), output.end());
     return output;
}

