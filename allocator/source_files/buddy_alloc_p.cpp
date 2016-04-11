// buddy_alloc_p.cpp
#ifndef BUDDY_ALLOC
#define BUDDY_ALLOC

#include "Free_List.cpp"
#include "Handle.cpp"

#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <limits>
#include <iterator>
#include <list>
#include <mutex>
#include <utility>
#include <iostream>

namespace Flamingo{
namespace Memory {

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

     typedef pointer*				handle_ptr;
     typedef T*					pointer_in;
     typedef Free_List<T>			free_list;

     typedef std::pair<pointer_in, free_list*> tuple;
     typedef std::list<tuple>			bank;
     typedef typename bank::iterator	iter;

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
#include "buddy_alloc_p.inl"
}//end Memory
}//end Flamingo
#endif
