#ifndef standard_alloc
#define standard_alloc
#include <memory>
#include "location.cu"

namespace Flamingo{
namespace Memory{

template <typename T, typename Policy>
class standard_alloc_policy : public Policy {
    public:
     typedef Policy Location_Policy;

    public:
     //    typedefs
     typedef T					value_type;
     typedef value_type*			pointer;
     typedef const value_type*	const_pointer;
     typedef value_type&			reference;
     typedef const value_type&	const_reference;
     typedef std::size_t			size_type;
     typedef std::ptrdiff_t difference_type;

    public:
     //    convert an StandardAllocPolicy<T> to StandardAllocPolicy<U>
     template <typename U, typename P>
     struct rebind {
          typedef standard_alloc_policy<U, P> other;
     };

    public:
     inline explicit standard_alloc_policy() {
     }
     inline ~standard_alloc_policy() {
     }
     inline explicit standard_alloc_policy(standard_alloc_policy const&) {
     }
     template <typename U, typename P>
     inline explicit standard_alloc_policy(standard_alloc_policy<U, P> const&) {
     }

     //    memory allocation
     inline pointer allocate(size_type size) {
          return reinterpret_cast<pointer>(Policy::New(size));
     }
     inline void deallocate(pointer p) {
          Policy::Delete(p);
     }

     inline void addbank(size_type s) {};

     //    size
     inline size_type max_size() const {
          return std::numeric_limits<size_type>::max();
     }
};  
//    end of class StandardAllocPolicy

// determines if memory from another
// allocator can be deallocated from this one
// template<typename T, typename T2>
// inline bool operator==(StandardAllocPolicy<T> const&,
//                       StandardAllocPolicy<T2> const&) {
//    return true;
//}
// template<typename T, typename OtherAllocator>
// inline bool operator==(StandardAllocPolicy<T> const&,
//                                    OtherAllocator const&) {
//    return false;
//}
}//end Memory
}//end Flamingo
#endif
