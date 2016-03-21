// buddy_alloc_p.cpp
#ifndef BUDDY_ALLOC
#define BUDDY_ALLOC

#include \
    "Free_List.cpp"
#include \
    "Handle.cpp"

#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <limits>
#include <iterator>
#include <list>
#include <mutex>
#include <utility>
#include <iostream>

/** \ingroup allocator-module
 */
template <typename T, typename Policy>
class buddy_alloc_policy : public Policy {
    public:
     typedef Policy Location_Policy;

    public:
     //    typedefs
     typedef T					value_type;
     typedef Handle<T>			pointer;
     typedef Handle<const T>		const_pointer;
     typedef Handle_void			void_pointer;
     typedef const Handle_void	const_void_pointer;
     typedef value_type&			reference;
     typedef const value_type&	const_reference;
     typedef std::size_t			size_type;
     typedef typename Handle<T>::difference_type difference_type;


     typedef std::false_type propagate_on_container_copy_assignment;
     typedef std::false_type propagate_on_container_move_assignment;
     typedef std::false_type propagate_on_container_swap;
     typedef std::false_type is_allways_equal;

     typedef pointer* handle_ptr;
     typedef T* pointer_in;
     typedef Free_List<T> free_list;
     typedef std::pair<pointer_in, free_list*> tuple;
     typedef std::list<tuple> bank;
     typedef typename bank::iterator iter;

/*     template <typename U, typename L>
     struct rebind {
          typedef buddy_alloc_policy<U, L> other;
     };
*/
     static bank _bank;
     static std::mutex mutex;

     inline buddy_alloc_policy();
     buddy_alloc_policy(buddy_alloc_policy<T, Policy>&);
     buddy_alloc_policy(const buddy_alloc_policy<T, Policy>&);

     void addbank(iter&);
	void removebank(iter&); 

     //    memory allocation
     pointer allocate(size_type);
     void deallocate(pointer);
     void deallocate(size_type, pointer);
     void deallocate(pointer, size_type);

     //    size
     inline size_type max_size() const {
          return std::numeric_limits<size_type>::max();
     }
};
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
buddy_alloc_policy<T, Policy>::pointer buddy_alloc_policy<T, Policy>::allocate(
    buddy_alloc_policy<T, Policy>::size_type size) {
     auto it = _bank.begin();
     handle_ptr free_handle = (it->second)->find_free_handle(size);
     int n = 0;
     while (!free_handle) {
          n++;
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
                      [&p](tuple t) { return (t.first) == (p._base_pointer); });

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
#endif
