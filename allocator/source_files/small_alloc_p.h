#ifndef small_alloc
#define small_alloc

/** \ingroup allocator-module
 */
template <typename T, typename Policy>
class small_alloc_policy : public Policy {
    private:
     typedef Policy Location_Policy;

    public:
     //    typedefs
     typedef T value_type;
     typedef value_type* pointer;
     typedef const value_type* const_pointer;
     typedef value_type& reference;
     typedef const value_type& const_reference;
     typedef std::size_t size_type;
     typedef std::ptrdiff_t difference_type;

    public:
     //    convert an StandardAllocPolicy<T> to StandardAllocPolicy<U>
     template <typename U>
     struct rebind {
          typedef small_alloc_policy<U> other;
     };

    public:
     inline explicit small_alloc_policy() {
     }
     inline ~small_alloc_policy() {
     }
     inline explicit small_alloc_policy(small_alloc_policy const&) {
     }
     template <typename U>
     inline explicit small_alloc_policy(small_alloc_policy<U> const&) {
     }

    public:
     //    memory allocation
     inline pointer allocate(size_type cnt,
                             typename std::allocator<void>::const_pointer = 0) {
     }
     inline void deallocate(pointer p, size_type) {
     }

     //    size
     inline size_type max_size() const {
          return std::numeric_limits<size_type>::max();
     }
};  //    end of class StandardAllocPolicy

// determines if memory from another
// allocator can be deallocated from this one
template <typename T, typename T2>
inline bool operator==(buddy_alloc_policy<T> const&,
                       small_alloc_policy<T2> const&) {
     return true;
}
template <typename T, typename OtherAllocator>
inline bool operator==(small_alloc_policy<T> const&, OtherAllocator const&) {
     return false;
}

#endif
