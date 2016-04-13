#include "reference.cu"

//---------------------host------------------------------------
template<typename T,Memory::Region M>
reference_wrapper<T,M>& 
	reference_wrapper<T,M>::operator=(const reference_wrapper<T,M>& other)
{
	if(!_ptr){
		_ptr=static_cast<T*>(Memory::location<M>::New(sizeof(T)) ); 
	}
	Memory::location<M>::MemCopy(	other._ptr,
							_ptr,
							sizeof(T)); 
	return *this; 
}

template<typename T,Memory::Region M>
reference_wrapper<T,M>& 
	reference_wrapper<T,M>::operator=(const T& x)
{
	*_ptr=x; 
	return *this; 
}

template<typename T,Memory::Region M>
T& reference_wrapper<T,M>::get()const
{
	return *_ptr; 
}
//-----------------------device-------------------------------
template<typename T>
reference_wrapper<T,Memory::Region::device>& 
	reference_wrapper<T,Memory::Region::device>::operator=(const reference_wrapper<T,Memory::Region::device>& other)
{
	typedef Memory::location<Memory::Region::device> Location; 

	if(!_ptr){
		_ptr=static_cast<T*>(Location::New(sizeof(T)) ); 
	}
	Location::MemCopy(	other._ptr,
					_ptr,
					sizeof(T)); 
	return *this; 
};

template<typename T>
reference_wrapper<T,Memory::Region::device>& 
	reference_wrapper<T,Memory::Region::device>::operator=(const T& x)
{
	#ifdef __CUDA_ARCH__
		*_ptr=x;  
	#else
		T copy=x;
		Memory::location<Memory::Region::device>::
			MemCopy(	&copy,
					_ptr,
					sizeof(T));
	#endif
	return *this; 
}

template<typename T>
const T& reference_wrapper<T,Memory::Region::device>::get()const
{
	#ifdef __CUDA_ARCH__
		return *_ptr; 	
	#else
		T* host_ptr=const_cast<T*>(&_value_host);
		Memory::location<Memory::Region::device>::
			MemCopy(	_ptr,
					host_ptr,
					sizeof(T)); 
		return _value_host; 
	#endif
}



















