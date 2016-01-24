#ifndef object_traits
#define object_traits

template <typename T>
class ObjectTraits {
    public:
     //    convert an ObjectTraits<T> to ObjectTraits<U>
     template <typename U>
     struct rebind {
          typedef ObjectTraits<U> other;
     };

    public:
     inline explicit ObjectTraits() {
     }
     inline ~ObjectTraits() {
     }
     template <typename U>
     inline explicit ObjectTraits(ObjectTraits<U> const&) {
     }

     //    address
     inline T* address(T& r) {
          return &r;
     }
     inline T const* address(T const& r) {
          return &r;
     }

     inline void construct(T* p, const T& t) {
          new (p) T(t);
     }
     inline void destroy(T* p) {
          p->~T();
     }
};  //    end of class ObjectTraits

#endif
