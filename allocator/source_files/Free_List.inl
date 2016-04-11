// free_list.inl
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
	//true  - free container is not empty 
	//false - free container is empty
     return !(free_container.empty());
}
template <typename T>
bool Free_List<T>::allFree() {
	//true  - reserved container is empty 
	//false - reserved container is not empty
     return (reserved_container.empty());
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
          pointer buddy = new Handle<T>(	p->buddy_offset(),	
									p->_size, 
									p->_base_pointer
								);

          free_container.insert(buddy);
          split(p, size);
     }
}

template <typename T>
Free_List<T>::pointer Free_List<T>::find_free_handle(
    Free_List<T>::size_type size) {
     return free_container.find_remove_handle(size);
}
