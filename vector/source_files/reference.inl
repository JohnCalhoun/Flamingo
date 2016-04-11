#include "reference.cu"

//---------------------host------------------------------------
template<typename T,Memory M>
reference_wrapper<T,M>& 
	reference_wrapper<T,M>::operator=(const reference_wrapper<T,M>& other)
{
	_ptr=other._ptr; 	
	return *this; 
}

template<typename T,Memory M>
reference_wrapper<T,M>& 
	reference_wrapper<T,M>::operator=(const T& x)
{
	*_ptr=x; 
	return *this; 
}

template<typename T,Memory M>
T& reference_wrapper<T,M>::get()const
{
	return *_ptr; 
}
//-----------------------device-------------------------------
template<typename T>
reference_wrapper<T,device>& 
	reference_wrapper<T,device>::operator=(const reference_wrapper<T,device>& other)
{
	_ptr=other.ptr; 
	return *this; 
};

template<typename T>
reference_wrapper<T,device>& 
	reference_wrapper<T,device>::operator=(const T& x)
{
	#ifdef ___CUDA_ARCH___
		*_ptr=x;  
	#else
		T copy=x;
		location<device>::MemCopy(&copy,_ptr,sizeof(T));
	#endif
	return *this; 
}

template<typename T>
T& reference_wrapper<T,device>::get()const
{
	#ifdef ___CUDA_ARCH__
		return *_ptr; 	
	#else
		location<device>::MemCopy(_ptr,_ptr_host,sizeof(T)); 
		return *_ptr_host; 
	#endif
}



















