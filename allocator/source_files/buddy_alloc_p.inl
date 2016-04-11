// buddy_alloc_p.inl

template <typename T, typename P>
buddy_alloc_policy<T, P>::bank buddy_alloc_policy<T, P>::_bank;

template <typename T, typename P>
std::mutex buddy_alloc_policy<T, P>::mutex;

//    end of class StandardAllocPolicy
//***************************************operator
// overloads***********************************
// determines if memory from another
// allocator can be deallocated from this one
template <typename T, typename T2, typename P>
inline bool operator==(buddy_alloc_policy<T, P> const&,
                       buddy_alloc_policy<T2, P> const&) {
     return false;
}
template <typename T, typename P>
inline bool operator==(buddy_alloc_policy<T, P> const&,
                       buddy_alloc_policy<T, P> const&) {
     return false;
}
template <typename T, typename T2, typename P, typename P2>
inline bool operator==(buddy_alloc_policy<T, P> const&,
                       buddy_alloc_policy<T2, P2> const&) {
     return false;
}

template <typename T, typename P, typename OtherAllocator>
inline bool operator==(buddy_alloc_policy<T, P> const&, OtherAllocator const&) {
     return false;
}

template <typename T, typename P, typename T2, typename P2>
inline bool operator!=(buddy_alloc_policy<T, P> const& b1,
                       buddy_alloc_policy<T2, P2> const& b2) {
     return !(b1 == b2);
}

template <typename T, typename P, typename OtherAllocator>
inline bool operator!=(buddy_alloc_policy<T, P> const&, OtherAllocator const&) {
     return true;
}

//**********************************operator
// overloads****************************************
//*****************************constructor/destructor*******************************************
template <typename T, typename Policy>
buddy_alloc_policy<T, Policy>::buddy_alloc_policy() {
     if (_bank.empty()) {
          iter it = _bank.begin();
          addbank(it);
     }
}
template <typename T, typename Policy>
buddy_alloc_policy<T, Policy>::buddy_alloc_policy(
    buddy_alloc_policy<T, Policy>& other) {}
template <typename T, typename Policy>
buddy_alloc_policy<T, Policy>::buddy_alloc_policy(
    const buddy_alloc_policy<T, Policy>& other) {}

template <typename T, typename Policy>
void buddy_alloc_policy<T, Policy>::addbank(
    buddy_alloc_policy<T, Policy>::iter& it) {
     if (it == _bank.end()) {
          std::lock_guard<std::mutex> lock(mutex);
          if (it == _bank.end()) {
               int new_size = _bank.size() + 2;
               int size = std::exp2(new_size) * sizeof(T);
               pointer_in base_pointer = static_cast<pointer_in>(Policy::New(size));
               handle_ptr first_handle = new pointer(0, size, base_pointer);
               _bank.push_back(std::make_pair(base_pointer, new free_list()));
               ((_bank.back()).second)->add_free_handle(first_handle);
               it--;
          }
     }
}

template <typename T, typename Policy>
void buddy_alloc_policy<T, Policy>::removebank(
    buddy_alloc_policy<T, Policy>::iter& it) {
     if (it == _bank.end()) {
          std::lock_guard<std::mutex> lock(mutex);
          if (it == _bank.end()) {
			if( (it->second)->allFree() ){
				Policy::Delete(it->first);
				delete(it->second);
				_bank.pop_back(); 				
			}
		}
     }
}

//************************************************allocate/deallocate***********************************************
// allocate
template <typename T, typename Policy>
buddy_alloc_policy<T, Policy>::pointer 
	buddy_alloc_policy<T, Policy>::allocate(
		buddy_alloc_policy<T, Policy>::size_type size) 
{
	if(size==1){
		size++; //nasty fix
	}
     auto it = _bank.begin();
     handle_ptr free_handle = (it->second)->find_free_handle(size);
     while (!free_handle) {
          it++;
          addbank(it);
          free_handle = (it->second)->find_free_handle(size);
     }
     (it->second)->split(free_handle, size);
     (it->second)->add_reserved_handle(free_handle);
     return (*free_handle);
}

// deallocate
template <typename T, typename Policy>
void buddy_alloc_policy<T, Policy>::deallocate(
			buddy_alloc_policy<T, Policy>::pointer p) {
     // search for matching base pointer
     auto it = std::find_if(_bank.begin(), _bank.end(),
                      [&p](tuple t) { 
					return (t.first) == (p._base_pointer); 
					}
					);

     int offset = p._offset;
     handle_ptr handle = (it->second)->find_reserved_handle(offset);
     (it->second)->combine(handle);
     (it->second)->add_free_handle(handle);
	removebank(it); 		
}

template <typename T, typename Policy>
void buddy_alloc_policy<T, Policy>::deallocate(
    buddy_alloc_policy<T, Policy>::size_type size,
    buddy_alloc_policy<T, Policy>::pointer p) {
     deallocate(p);
}
template <typename T, typename Policy>
void buddy_alloc_policy<T, Policy>::deallocate(
    buddy_alloc_policy<T, Policy>::pointer p,
    buddy_alloc_policy<T, Policy>::size_type size) {
     deallocate(p);
}

//***************************************allcoate/deallocate*************************************************************
//***************************************utilities********************************************************************
template <typename T, typename Policy>
std::ostream& operator<<(std::ostream& out,
                         buddy_alloc_policy<T, Policy>& alloc) {
     std::for_each((alloc._bank).begin(), (alloc._bank).end(),
                   [&out](typename buddy_alloc_policy<T, Policy>::tuple t) {
    out << "Free List" << '\n';
    out << *(t.second) << '\n';
     });
     return out;
}

//*************************************utilities************************************************************************
