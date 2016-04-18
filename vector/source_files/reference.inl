#include "reference.cu"

//---------------------host------------------------------------
template<typename T>
reference_wrapper<T>& 
	reference_wrapper<T>::operator=(const reference_wrapper<T>& other)
{
	_ptr=other._ptr; 
	return *this; 
}

template<typename T>
reference_wrapper<T>& 
	reference_wrapper<T>::operator=(const T& x)
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
bool reference_wrapper<T>::operator==(reference_wrapper<T>& other)
{
	return get()==other.get();
}

template<typename T>
T reference_wrapper<T>::get()const
{
	#ifdef __CUDA_ARCH__
		return *_ptr; 	
	#else
		T _value_host;
		Memory::location<Memory::Region::device>::
			MemCopy(	_ptr,
					&_value_host,
					sizeof(T)); 
		return _value_host; 
	#endif
}




