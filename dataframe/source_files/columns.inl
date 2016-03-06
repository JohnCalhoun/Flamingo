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
	_location=other.getlocation(); 
	switch(_location)
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
template<typename T>
column<T>::size_type column<T>::size()const
{
	size_type size; 
	switch(getlocation())
	{
		case host:
		{
			host_column* host_ptr= static_cast<host_column*>(_ptr); 
			size=host_ptr->size(); 
			break; 
		}
		case device:
		{
			device_column* device_ptr=static_cast<device_column*>(_ptr); 
			size=device_ptr->size(); 
			break; 
		}
		case pinned:
		{
			pinned_column* pinned_ptr=static_cast<pinned_column*>(_ptr); 
			size=pinned_ptr->size(); 
			break; 
		}
		case unified:
		{
			unified_column* unified_ptr=static_cast<unified_column*>(_ptr); 
			size=unified_ptr->size(); 
			break; 
		}

	}
	return size; 
}

template<typename T>
column<T>::size_type column<T>::max_size()const
{
	size_type max_size; 
	switch(getlocation())
	{
		case host:
		{
			host_column* host_ptr= static_cast<host_column*>(_ptr); 
			max_size=host_ptr->max_size(); 
			break; 
		}
		case device:
		{
			device_column* device_ptr=static_cast<device_column*>(_ptr); 
			max_size=device_ptr->max_size(); 
			break; 
		}
		case pinned:
		{
			pinned_column* pinned_ptr=static_cast<pinned_column*>(_ptr); 
			max_size=pinned_ptr->max_size(); 
			break; 
		}
		case unified:
		{
			unified_column* unified_ptr=static_cast<unified_column*>(_ptr); 
			max_size=unified_ptr->max_size(); 
			break; 
		}

	}
	return max_size; 
}

template<typename T>
bool column<T>::empty()const{
	bool empty; 
	switch(getlocation())
	{
		case host:
		{
			host_column* host_ptr= static_cast<host_column*>(_ptr); 
			empty=host_ptr->empty(); 
			break; 
		}
		case device:
		{
			device_column* device_ptr=static_cast<device_column*>(_ptr); 
			empty=device_ptr->empty(); 
			break; 
		}
		case pinned:
		{
			pinned_column* pinned_ptr=static_cast<pinned_column*>(_ptr); 
			empty=pinned_ptr->empty(); 
			break; 
		}
		case unified:
		{
			unified_column* unified_ptr=static_cast<unified_column*>(_ptr); 
			empty=unified_ptr->empty(); 
			break; 
		}

	}
	return empty;
}
/*
template<typename T>
void column<T>::reserve(size_type)const{

}

template<typename T>
column<T>::size_type column<T>::capacity()const{

}
*/
template<typename T>
void column<T>::fill(T t){ 
	size_type s=size(); 
	switch(getlocation())
	{
		case host:
		{
			host_column* host_ptr= static_cast<host_column*>(_ptr); 
			host_ptr->assign(s,t); 			
			break; 
		}
		case device:
		{
			device_column* device_ptr=static_cast<device_column*>(_ptr); 
			device_ptr->assign(s,t); 				
			break; 
		}
		case pinned:
		{
			pinned_column* pinned_ptr=static_cast<pinned_column*>(_ptr); 
			pinned_ptr->assign(s,t); 				
			break; 
		}
		case unified:
		{
			unified_column* unified_ptr=static_cast<unified_column*>(_ptr); 
			unified_ptr->assign(s,t); 						
			break; 
		}

	}
}

template<typename T>
template<typename iter>
void column<T>::copy(iter start, iter stop){
	switch(getlocation())
	{
		case host:
		{
			host_column* host_ptr= static_cast<host_column*>(_ptr); 
			host_ptr->assign(start,stop);
			break; 
		}
		case device:
		{
			device_column* device_ptr=static_cast<device_column*>(_ptr); 
			device_ptr->assign(start,stop);
			break; 
		}
		case pinned:
		{
			pinned_column* pinned_ptr=static_cast<pinned_column*>(_ptr); 
			pinned_ptr->assign(start,stop);
			break; 
		}
		case unified:
		{
			unified_column* unified_ptr=static_cast<unified_column*>(_ptr); 
			unified_ptr->assign(start,stop);
			break; 
		}

	}
}


template<typename T>
void column<T>::clear(){
	switch(getlocation())
	{
		case host:
		{
			host_column* host_ptr= static_cast<host_column*>(_ptr); 
			host_ptr->clear();
			break; 
		}
		case device:
		{
			device_column* device_ptr=static_cast<device_column*>(_ptr); 
			device_ptr->clear();
			break; 
		}
		case pinned:
		{
			pinned_column* pinned_ptr=static_cast<pinned_column*>(_ptr); 
			pinned_ptr->clear();
			break; 
		}
		case unified:
		{
			unified_column* unified_ptr=static_cast<unified_column*>(_ptr); 
			unified_ptr->clear();
			break; 
		}

	}
}

#undef DEFAULT









