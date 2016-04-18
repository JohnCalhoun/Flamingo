#ifndef REFERENCE_VECTOR_CPP
#define REFERENCE_VECTOR_CPP

#include <location.cu>
#include <memory>
#include <utility>

namespace Flamingo{
namespace Vector{ 
#define __both__ __host__ __device__ 

//reference to memory on device
template <class T>
class reference_wrapper{
	public:
	// construct/copy/destroy
	__both__ reference_wrapper():_ptr(NULL){};  
	__both__ reference_wrapper(T& ref): _ptr(&ref){};
	__both__ reference_wrapper(T&&) = delete;
	__both__ reference_wrapper(const reference_wrapper& wrap):
		_ptr(wrap._ptr){};
	__both__ reference_wrapper(T* ptr):
		_ptr(ptr){}; 

	// assignment
	__both__ reference_wrapper& operator=(const reference_wrapper&);
	__both__ reference_wrapper& operator=(const T&);
	__both__ bool operator==(reference_wrapper<T>&);

	__both__ operator T ()const{ return get(); };

	__both__ T get()const;
	
	T* _ptr;
};

#include "reference.inl"
}
}
#undef __both__

#endif

