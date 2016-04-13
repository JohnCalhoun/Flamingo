// Handle.inl
#include "Handle.cpp"
namespace Flamingo {
namespace Memory {

 template <typename T>
__both__ Handle<T>::Handle(const Handle<T>& obj) {
     _base_pointer = obj._base_pointer;
     _offset = obj._offset;
     _size = obj._size;
};

template <typename T>
void Handle<T>::combine(const Handle<T>& rhs) {
     int new_off = std::min(rhs._offset, _offset);
     _offset = new_off;
     _size += rhs._size;
};

template <typename T>
int Handle<T>::buddy_offset() {
     int tag = _offset / _size;
     int dif;
     if (tag % 2 == 0) {
          dif = _size;
     } else {
          dif = -_size;
     }
     return _offset + dif;
}

template <typename T>
__both__ T& Handle<T>::operator*() {
     return *(_base_pointer + _offset/sizeof(T));
};

template <typename T>
__both__ T& operator*(const Handle<T>& handle) {
     return *(handle._base_pointer + handle._offset/sizeof(T));
};

template <typename T>
__both__ T* Handle<T>::operator->() {
     return _base_pointer + _offset/sizeof(T);
};

// outofclass operators
template <typename T>
__both__ Handle<T> operator+(const Handle<T> handle, const int x) {
     Handle<T> temp = handle;
     int m;
     m = x*sizeof(T);//this can be rewriten
	if (m > 0) {
          while (m--)
               ++temp._offset;
     } else {
          while (m++)
               --temp._offset;
     }
     return temp;
}

template <typename T>
__both__ Handle<T> operator+(const int x, const Handle<T> handle) {
     return handle + x;
}

template <typename T>
__both__ int operator-(const Handle<T>& handle_1, const Handle<T>& handle_2) {
     return (handle_1._offset - handle_2._offset)/sizeof(T);
}

template <typename T>
__both__ Handle<T> operator-(const Handle<T>& handle, const int& x) {
     Handle<T> temp = handle;
     temp._offset -= x*sizeof(T);
     return temp;
}

// inclass operators
template <typename T>
Handle<T>& Handle<T>::operator=(const Handle<T>& handle) {
     this->_offset = handle._offset;
     this->_size = handle._size;
     this->_base_pointer = static_cast<T*>(handle._base_pointer);
     return *this;
}

template <typename T>
bool Handle<T>::operator==(const Handle<T>& handle_2) const {
     return this->_offset == handle_2._offset;
}

template <typename T, typename U>
__both__ bool operator==(const Handle<T>& handle_1, const Handle<U>& handle_2) {
     return false;
}

template <typename T>
bool operator==(const Handle<T>& handle_1, std::nullptr_t& null) {
     return handle_1._base_pointer == null;
}

template <typename T>
bool operator==(std::nullptr_t& null, const Handle<T>& handle_1) {
     return handle_1._base_pointer == null;
}

template <typename T>
bool Handle<T>::operator!=(const Handle<T>& handle) const {
     return !((*this) == handle);
}
template <typename T>
bool Handle<T>::operator!=(const std::nullptr_t& null) const {
     return this->_base_pointer != null;
}

template <typename T>
bool operator!=(const std::nullptr_t& null, const Handle<T>& handle_1) {
     return handle_1._base_pointer != null;
}

template <typename T>
template <typename U>
bool Handle<T>::operator!=(const Handle<U>& handle_2) const {
     return !((*this) == handle_2);
}

// random access iterator
template <typename T>
T& Handle<T>::operator[](const int& x) {
     return *(_base_pointer + x);
}

template <typename T>
bool Handle<T>::operator<(const Handle<T>& handle_2) const {
     return (this->_base_pointer+this->_offset)< (handle_2._base_pointer+handle_2._offset);
}

template <typename T>
bool Handle<T>::operator>(const Handle<T>& handle_2) const {
     return handle_2<(*this);
}

template <typename T>
bool Handle<T>::operator<=(const Handle<T>& handle_2) const {
     return !((*this) > handle_2);
}

template <typename T>
bool Handle<T>::operator>=(const Handle<T>& handle_2) const {
     return !((*this) < handle_2);
}

template <typename T>
Handle<T>& Handle<T>::operator--() {
	_offset-=sizeof(T); 
     return *this;
}

template <typename T>
Handle<T>& Handle<T>::operator++() {
	_offset+=sizeof(T);	
     return *this;
}

template <typename T>
Handle<T> Handle<T>::operator--(int x) {
     Handle<T> temp = *this;
	_offset-=sizeof(T); 
     return temp;
}

template <typename T>
Handle<T> Handle<T>::operator++(int x) {
     Handle<T> temp = *this;
	_offset+=sizeof(T); 
     return temp;
}

template <typename T>
Handle<T>& Handle<T>::operator-=(const int x) {
     this->_offset -= x*sizeof(T);
     return *this;
}

template <typename T>
Handle<T>& Handle<T>::operator+=(const int x) {
     this->_offset += x*sizeof(T);
     return *this;
}

// casting
template <typename T>
template <typename U>
Handle<T>::operator Handle<U>()const {
     Handle<U> tmp;
	tmp._offset		=this->_offset;
	tmp._size			=this->_size;
	tmp._base_pointer	=this->_base_pointer;
     return tmp;
}
// use type safe bool idiom
template <typename T>
Handle<T>::operator bool_type() const {
     return (_base_pointer)
                ? &Handle<T>::this_type_does_not_support_comparisons
                : 0;
}

template <typename T>
template <typename L>
Handle<T>::operator L*() {
     return (L*)((this->_base_pointer) + (this->_offset/sizeof(T)));
}

//****************************void handle*****************
Handle_void::Handle_void(const Handle_void& other)
    : _offset(other._offset), _size(other._size), _base_pointer(other._base_pointer) {};

template <typename T>
Handle_void::Handle_void(const Handle<T>& handle)
    : _offset(handle._offset), _size(handle._size) {
     _base_pointer = static_cast<void*>(handle._base_pointer);
};

template <typename T>
Handle_void::Handle_void(T* p)
    : _offset(0), _size(0) {
     _base_pointer = p;
};

Handle_void::Handle_void(std::nullptr_t)
    : _base_pointer(NULL), _offset(0), _size(0) {};

__both__ Handle_void& Handle_void::operator=(const Handle_void& handle) {
     this->_base_pointer = NULL;
     this->_offset = 0;
     this->_size = handle._size;
     return *this;
};

Handle_void& Handle_void::operator=(std::nullptr_t& null) {
     this->_base_pointer = NULL;
     this->_offset = 0;
     this->_size = 0;
     return *this;
};
//*************out of member operators
template <typename T>
__both__ Handle_void::operator Handle<T>() {
     Handle<T> temp = *this;
     return temp;
};

template <typename T>
__both__ Handle_void::operator T*() {
     return (T*)(this->_base_pointer);
};

}
}
