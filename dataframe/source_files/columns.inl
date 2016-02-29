//columns.inl
#include"columns.cpp"
#define DEFAULT_LOCATION host
#define DEFAULT_COLUMN host_column

template<typename T>
column<T>::column(){
	DEFAULT_COLUMN* tmp=new DEFAULT_COLUMN();	
	_ptr=static_cast<void*>(tmp); 
	_location=DEFAULT_LOCATION;
}

template<typename T>
column<T>::column(int n){
	DEFAULT_COLUMN* tmp=new DEFAULT_COLUMN(n);	
	_ptr=static_cast<void*>(tmp); 
	_location=DEFAULT_LOCATION;
}

template<typename T>
column<T>::column(const column<T>& other){
	switch(getlocation())
	{
		case host:
		{
			delete static_cast<host_column*>(_ptr);
			break; 
		}
		case device:
		{
			delete static_cast<device_column*>(_ptr); 
			break; 
		}
		case pinned:
		{
			delete static_cast<pinned_column*>(_ptr); 
			break; 
		}
		case unified:
		{
			delete static_cast<unified_column*>(_ptr); 
			break; 
		}

	}
	switch(other.getlocation())
	{
		case host:
		{
			*static_cast<host_column*>(_ptr)=*static_cast<host_column*>(other._ptr); 
			break; 
		}
		case device:
		{
			*static_cast<device_column*>(_ptr)=*static_cast<device_column*>(other._ptr); 
			break; 
		}
		case pinned:
		{
			*static_cast<pinned_column*>(_ptr)=*static_cast<pinned_column*>(other._ptr); 
			break; 
		}
		case unified:
		{
			*static_cast<unified_column*>(_ptr)=*static_cast<unified_column*>(other._ptr); 
			break; 
		}
	}
}

template<typename T>
column<T>::~column(){
	switch(getlocation())
	{
		case host:
		{
			delete static_cast<host_column*>(_ptr); 
			break; 
		}
		case device:
		{
			delete static_cast<device_column*>(_ptr); 
			break; 
		}
		case pinned:
		{
			delete static_cast<pinned_column*>(_ptr); 
			break; 
		}
		case unified:
		{
			delete static_cast<unified_column*>(_ptr); 
			break; 
		}

	}
} 

template<typename T>
template<typename Aloc>
column<T>::column(const thrust::device_vector<T,Aloc>& other){
	device_column* tmp=new device_column();
	*tmp=other;
	_ptr=static_cast<void*>(tmp); 
	_location=device;
} 

template<typename T>
template<typename Aloc>
column<T>::column(const thrust::host_vector<T,Aloc>& other){
	host_column* tmp=new host_column();
	*tmp=other;
	_ptr=static_cast<void*>(tmp); 
	_location=host;
} 

template<typename T>
template<Memory M>
void column<T>::move(){
	if( M != _location){
		typedef typename Return<M>::raw new_col;
		new_col* tmp=new new_col(); 
		
		switch(getlocation())
		{
			case host:
			{
				host_column* base=static_cast<host_column*>(_ptr); 
				*tmp=*base;
				delete base;
				_ptr=static_cast<void*>(tmp);				
				break;
			}
			case device:
			{
				host_column* base=static_cast<device_column*>(_ptr); 
				*tmp=*base;
				delete base;
				_ptr=static_cast<void*>(tmp);				
				break;
			}
			case pinned:
			{
				host_column* base=static_cast<pinned_column*>(_ptr); 
				*tmp=*base;
				delete base;
				_ptr=static_cast<void*>(tmp);				
				break;
			}

			case unified:
			{
				host_column* base=static_cast<unified_column*>(_ptr); 
				*tmp=*base;
				delete base;
				_ptr=static_cast<void*>(tmp);				
				break;
			}

		}
	}
}
template<typename T>
Memory column<T>::getlocation()const{
	return _location; 
} 

template<typename T>
template<Memory M>
typename column<T>::Return<M>::type column<T>::access(){
	return static_cast<typename Return<M>::type>(access_raw()); 
} 

template<typename T>
void* column<T>::access_raw(){
	return _ptr; 
} 

template<typename T>
void column<T>::swap(column<T>& other ){
	std::swap(_location,other._location);
	std::swap(_ptr,other._ptr);
} 

template<typename T>
column<T>& column<T>::operator=(const column<T>& other){
	column<T> tmp(other);
	swap(tmp);
	return *this;
}

#undef DEFAULT













