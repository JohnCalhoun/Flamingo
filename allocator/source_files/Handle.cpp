// Handle.cpp
#ifndef HANDLE
#define HANDLE

#include <cstddef>
#include <type_traits>
#include <iterator>

#define __both__ __device__ __host__
namespace Flamingo {
namespace Memory {

template <typename T>
class Handle {
    public:
     typedef T value_type;
     typedef int difference_type;
     typedef T* pointer;
     typedef T& reference;

     typedef std::random_access_iterator_tag iterator_category;
     // suport type safe bool idiom
	  typedef void (Handle<T>::*bool_type)() const;
     __both__ void this_type_does_not_support_comparisons() const {};

     pointer	_base_pointer;
     int		_offset; //this is the byte offset
     size_t	_size;

     // Special members
     Handle(int offset, size_t size, T* base)
		: _offset(offset), _size(size), _base_pointer(base) {};
     __both__ Handle()
		: _offset(0),_size(0),_base_pointer(NULL){};
     __both__ Handle(std::nullptr_t): Handle(){};
     __both__ Handle(Handle<T>&&);
	__both__ Handle(const Handle<T>&);
     template<typename L>
	__both__ Handle(L* x)
		: _base_pointer(x),_offset(0),_size(0) {}	
	__both__ ~Handle() {};


     // utility functions
     int buddy_offset();
     void combine(const Handle<T>&);
     size_t size() {return _size;};
     // operator overloads,
     // basic pointer
     __both__ T& operator*();
     __both__ pointer operator->();
     // nullable pointer
	__both__ Handle<T>& operator=(const Handle<T>&);

     __both__ bool operator==(const Handle<T>&) const;
     __both__ bool operator!=(const Handle<T>&) const;
     __both__ bool operator!=(const std::nullptr_t&) const;
     template <typename U>
     __both__ bool operator!=(const Handle<U>&) const;
     // random access iterator
     __both__ T& operator[](const int&);
     __both__ bool operator<(const Handle<T>&) const;
     __both__ bool operator>(const Handle<T>&) const;
     __both__ bool operator<=(const Handle<T>&) const;
     __both__ bool operator>=(const Handle<T>&) const;
     __both__ Handle<T>& operator--();
     __both__ Handle<T>& operator++();
     __both__ Handle<T> operator--(int);
     __both__ Handle<T> operator++(int);
     __both__ Handle<T>& operator-=(const int);
     __both__ Handle<T>& operator+=(const int);

     // casting
     template <typename U>
     __both__ operator Handle<U>()const;
     __both__ operator bool_type() const;

     template <typename L>
     __both__ operator L*();
};
//****************************void handle*****************

// SFINE for specialazationf or void and const void
class Handle_void {
    public:
     void* _base_pointer;
     int _offset;
     size_t _size;

     __both__ Handle_void()
         : _base_pointer(NULL), _offset(0), _size(0) {};
     template <typename T>
     __both__ Handle_void(const Handle<T>&);
     template <typename T>
     __both__ Handle_void(T*);
     __both__ Handle_void(const Handle_void&);
     __both__ Handle_void(const std::nullptr_t);

     __both__ Handle_void& operator=(const Handle_void&);
     __both__ Handle_void& operator=(std::nullptr_t&);
     template <typename T>
     __both__ operator Handle<T>();

     template <typename T>
     __both__ operator T*();
};


}//namespace Memory 
}//namespace Flamingo

#include "traits.cpp"
#include "Handle.inl"


#undef __both__

#endif
