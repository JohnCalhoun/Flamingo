//columns.inl
#include"columns.cpp"
#include <tuple>
#include <thrust/device_ptr.h>
#define DEFAULT_LOCATION Memory::Region::host
#define DEFAULT_COLUMN Traits::host_column

template<typename T>
column<T>::column(){
	_location=DEFAULT_LOCATION;
}

template<typename T>
column<T>::column(int n){
	_location=DEFAULT_LOCATION; 
	(std::get<DEFAULT_COLUMN>(_tuple)).resize(n);  
}

template<typename T>
column<T>::column(const column<T>& other){
	_location=other.getlocation(); 
	_tuple=other._tuple; 
}

template<typename T>
column<T>::~column(){} 

template<typename T>
template<typename Aloc>
column<T>::column(const thrust::device_vector<T,Aloc>& other){
	const int destination=Memory2Type<Memory::Region::device>::type::value;
	std::get<destination>(_tuple)=other; 
	_location=Memory::Region::device;
} 

template<typename T>
template<typename Aloc>
column<T>::column(const thrust::host_vector<T,Aloc>& other){
	const int destination=Memory2Type<Memory::Region::host>::type::value;
	std::get<destination>(_tuple)=other; 
	_location=Memory::Region::host;
} 

template<typename T>
template<Memory::Region M>
void column<T>::move(){
	if( M != _location){
		switch(getlocation())
		{
			case Memory::Region::host:
			{
				const int source=Memory2Type<Memory::Region::host>::type::value;
				const int destination=Memory2Type<M>::type::value;

				std::get<destination>(_tuple)=std::get<source>(_tuple); 
				std::get<source>(_tuple).clear(); 
				break;
			}
			case Memory::Region::device:
			{
				const int source=Memory2Type<Memory::Region::device>::type::value;
				const int destination=Memory2Type<M>::type::value;

				std::get<destination>(_tuple)=std::get<source>(_tuple); 
				std::get<source>(_tuple).clear(); 
				break;
			}
			case Memory::Region::pinned:
			{
				const int source=Memory2Type<Memory::Region::pinned>::type::value;
				const int destination=Memory2Type<M>::type::value;

				std::get<destination>(_tuple)=std::get<source>(_tuple); 
				std::get<source>(_tuple).clear(); 
				break;
			}
			case Memory::Region::unified:
			{
				const int source=Memory2Type<Memory::Region::unified>::type::value;
				const int destination=Memory2Type<M>::type::value;

				std::get<destination>(_tuple)=std::get<source>(_tuple); 
				std::get<source>(_tuple).clear(); 
				break;
			}
		}//end switch
		_location=M; 
	}//end if
}
template<typename T>
Memory::Region column<T>::getlocation()const{
	return _location; 
} 

template<typename T>
template<Memory::Region M>
typename column<T>::Return<M>::column& column<T>::access(){
	const int position=Memory2Type<M>::type::value;
	return std::get<position>(_tuple); 
} 
template<typename T>
column<T>::pointer column<T>::data(){
	pointer ptr;	
	switch(getlocation())
	{
			case Memory::Region::host:
			{
				const int position=Memory2Type<Memory::Region::host>::type::value;
				ptr=std::get<position>(_tuple).begin();
				break;
			}
			case Memory::Region::device:
			{
				const int position=Memory2Type<Memory::Region::device>::type::value;
				ptr=std::get<position>(_tuple).begin();
				break;
			}
			case Memory::Region::pinned:
			{
				const int position=Memory2Type<Memory::Region::pinned>::type::value;
				ptr=std::get<position>(_tuple).begin();
				break;
			}
			case Memory::Region::unified:
			{
				const int position=Memory2Type<Memory::Region::unified>::type::value;
				ptr=std::get<position>(_tuple).begin();
				break;
			}
	}
	return ptr; 
} 
template<typename T>
column<T>::const_pointer column<T>::data()const{
	switch(getlocation())
	{
			case Memory::Region::host:
			{
				const int position=Memory2Type<Memory::Region::host>::type::value;
				return std::get<position>(_tuple).begin();
			}
			case Memory::Region::device:
			{
				const int position=Memory2Type<Memory::Region::device>::type::value;
				return std::get<position>(_tuple).begin();
			}
			case Memory::Region::pinned:
			{
				const int position=Memory2Type<Memory::Region::pinned>::type::value;
				return std::get<position>(_tuple).begin();
			}
			case Memory::Region::unified:
			{
				const int position=Memory2Type<Memory::Region::unified>::type::value;
				return std::get<position>(_tuple).begin();
			}
			default:
			{
				return NULL; 
			}
			
	}
} 

template<typename T>
void column<T>::swap(column<T>& other ){
	std::swap(_location,other._location);
	std::swap(_tuple,other._tuple);
} 

template<typename T>
column<T>& column<T>::operator=(const column<T>& other){
	column<T> tmp(other);
	swap(tmp);
	return *this;
}
template<typename T>
bool column<T>::operator==(const column<T>& other)const{
	bool result; 
	if(getlocation()==other.getlocation()){
		result=(_tuple==other._tuple); 
	}else{
		result=false; 		
	}
	return result; 
}
template<typename T>
bool column<T>::operator!=(const column<T>& other)const{
	return !(*this==other);
}

template<typename T>
column<T>::size_type column<T>::size()const
{
	size_type size; 
	switch(getlocation())
	{
		case Memory::Region::host:
		{
			const int position=Memory2Type<Memory::Region::host>::type::value;
			size=std::get<position>(_tuple).size();
			break; 
		}
		case Memory::Region::device:
		{
			const int position=Memory2Type<Memory::Region::device>::type::value;
			size=std::get<position>(_tuple).size();
			break; 
		}
		case Memory::Region::pinned:
		{
			const int position=Memory2Type<Memory::Region::pinned>::type::value;
			size=std::get<position>(_tuple).size();
			break; 
		}
		case Memory::Region::unified:
		{
			const int position=Memory2Type<Memory::Region::unified>::type::value;
			size=std::get<position>(_tuple).size();
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
		case Memory::Region::host:
		{
			const int position=Memory2Type<Memory::Region::host>::type::value;
			max_size=std::get<position>(_tuple).max_size();
			break; 
		}
		case Memory::Region::device:
		{
			const int position=Memory2Type<Memory::Region::device>::type::value;
			max_size=std::get<position>(_tuple).max_size();
			break; 
		}
		case Memory::Region::pinned:
		{
			const int position=Memory2Type<Memory::Region::pinned>::type::value;
			max_size=std::get<position>(_tuple).max_size();
			break; 
		}
		case Memory::Region::unified:
		{
			const int position=Memory2Type<Memory::Region::unified>::type::value;
			max_size=std::get<position>(_tuple).max_size();
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
		case Memory::Region::host:
		{
			const int position=Memory2Type<Memory::Region::host>::type::value;
			empty=std::get<position>(_tuple).empty();
			break; 
		}
		case Memory::Region::device:
		{
			const int position=Memory2Type<Memory::Region::device>::type::value;
			empty=std::get<position>(_tuple).empty();
			break; 
		}
		case Memory::Region::pinned:
		{
			const int position=Memory2Type<Memory::Region::pinned>::type::value;
			empty=std::get<position>(_tuple).empty();
			break; 
		}
		case Memory::Region::unified:
		{
			const int position=Memory2Type<Memory::Region::unified>::type::value;
			empty=std::get<position>(_tuple).empty();
			break; 
		}
	}
	return empty;
}

template<typename T>
void column<T>::reserve(size_type size){
	switch(getlocation())
	{
		case Memory::Region::host:
		{
			const int position=Memory2Type<Memory::Region::host>::type::value;
			std::get<position>(_tuple).reserve(size);
			break; 
		}
		case Memory::Region::device:
		{
			const int position=Memory2Type<Memory::Region::device>::type::value;
			std::get<position>(_tuple).reserve(size);
			break; 
		}
		case Memory::Region::pinned:
		{
			const int position=Memory2Type<Memory::Region::pinned>::type::value;
			std::get<position>(_tuple).reserve(size);
			break; 
		}
		case Memory::Region::unified:
		{
			const int position=Memory2Type<Memory::Region::unified>::type::value;
			std::get<position>(_tuple).reserve(size);
			break; 
		}
	}
}

template<typename T>
column<T>::size_type column<T>::capacity()const{
	size_type size; 
	switch(getlocation())
	{
		case Memory::Region::host:
		{
			const int position=Memory2Type<Memory::Region::host>::type::value;
			size=std::get<position>(_tuple).capacity();
			break; 
		}
		case Memory::Region::device:
		{
			const int position=Memory2Type<Memory::Region::device>::type::value;
			size=std::get<position>(_tuple).capacity();
			break; 
		}
		case Memory::Region::pinned:
		{
			const int position=Memory2Type<Memory::Region::pinned>::type::value;
			size=std::get<position>(_tuple).capacity();
			break; 
		}
		case Memory::Region::unified:
		{
			const int position=Memory2Type<Memory::Region::unified>::type::value;
			size=std::get<position>(_tuple).capacity();
			break; 
		}
	}
	return size;
}

template<typename T>
void column<T>::fill(T t){ 
	size_type s=size(); 
	switch(getlocation())
	{
		case Memory::Region::host:
		{
			const int position=Memory2Type<Memory::Region::host>::type::value;
			std::get<position>(_tuple).assign(s,t);
			break; 
		}
		case Memory::Region::device:
		{
			const int position=Memory2Type<Memory::Region::device>::type::value;
			std::get<position>(_tuple).assign(s,t);
			break; 
		}
		case Memory::Region::pinned:
		{
			const int position=Memory2Type<Memory::Region::pinned>::type::value;
			std::get<position>(_tuple).assign(s,t);
			break; 
		}
		case Memory::Region::unified:
		{
			const int position=Memory2Type<Memory::Region::unified>::type::value;
			std::get<position>(_tuple).assign(s,t);
			break; 
		}
	}
}

template<typename T>
template<typename iter>
void column<T>::copy(iter start, iter stop){
	typedef typename Traits::host_column::iterator host_it;
	typedef typename Traits::device_column::iterator device_it; 
	switch(getlocation())
	{
		case Memory::Region::host:
		{
			const int position=Memory2Type<Memory::Region::host>::type::value;
			std::get<position>(_tuple).assign(start,stop);
			break; 
		}
		case Memory::Region::device:
		{
			const int position=Memory2Type<Memory::Region::device>::type::value;
			device_it ptr_start(start); 
			device_it ptr_stop(stop); 	
			std::get<position>(_tuple).assign(ptr_start,ptr_stop);
			break; 
		}
		case Memory::Region::pinned:
		{
			const int position=Memory2Type<Memory::Region::pinned>::type::value;
			device_it ptr_start(start); 
			device_it ptr_stop(stop); 	
			std::get<position>(_tuple).assign(ptr_start,ptr_stop);
			break; 
		}
		case Memory::Region::unified:
		{
			const int position=Memory2Type<Memory::Region::unified>::type::value;
			device_it ptr_start(start); 
			device_it ptr_stop(stop); 	
			std::get<position>(_tuple).assign(ptr_start,ptr_stop);
			break; 
		}
	}
}

template<typename T>
void column<T>::clear(){
	switch(getlocation())
	{
		case Memory::Region::host:
		{
			const int position=Memory2Type<Memory::Region::host>::type::value;
			std::get<position>(_tuple).clear();
			break; 
		}
		case Memory::Region::device:
		{
			const int position=Memory2Type<Memory::Region::device>::type::value;
			std::get<position>(_tuple).clear();
			break; 
		}
		case Memory::Region::pinned:
		{
			const int position=Memory2Type<Memory::Region::pinned>::type::value;
			std::get<position>(_tuple).clear();
			break; 
		}
		case Memory::Region::unified:
		{
			const int position=Memory2Type<Memory::Region::unified>::type::value;
			std::get<position>(_tuple).clear();
			break; 
		}
	}
}

template<typename T>
void column<T>::resize(column<T>::size_type n){
	switch(getlocation())
	{
		case Memory::Region::host:
		{
			const int position=Memory2Type<Memory::Region::host>::type::value;
			std::get<position>(_tuple).resize(n);
			break; 
		}
		case Memory::Region::device:
		{
			const int position=Memory2Type<Memory::Region::device>::type::value;
			std::get<position>(_tuple).resize(n);
			break; 
		}
		case Memory::Region::pinned:
		{
			const int position=Memory2Type<Memory::Region::pinned>::type::value;
			std::get<position>(_tuple).resize(n);
			break; 
		}
		case Memory::Region::unified:
		{
			const int position=Memory2Type<Memory::Region::unified>::type::value;
			std::get<position>(_tuple).resize(n);
			break; 
		}
	}
}

template<typename T>
void column<T>::resize(column<T>::size_type n,column<T>::value_type v){
	switch(getlocation())
	{
		case Memory::Region::host:
		{
			const int position=Memory2Type<Memory::Region::host>::type::value;
			std::get<position>(_tuple).resize(n,v);
			break; 
		}
		case Memory::Region::device:
		{
			const int position=Memory2Type<Memory::Region::device>::type::value;
			std::get<position>(_tuple).resize(n,v);
			break; 
		}
		case Memory::Region::pinned:
		{
			const int position=Memory2Type<Memory::Region::pinned>::type::value;
			std::get<position>(_tuple).resize(n,v);
			break; 
		}
		case Memory::Region::unified:
		{
			const int position=Memory2Type<Memory::Region::unified>::type::value;
			std::get<position>(_tuple).resize(n,v);
			break; 
		}
	}
}

template<typename T>
void column<T>::assign(
	column<T>::size_type s,
	column<T>::value_type v){
	switch(getlocation())
	{
		case Memory::Region::host:
		{
			const int position=Memory2Type<Memory::Region::host>::type::value;
			std::get<position>(_tuple).assign(s,v);
			break; 
		}
		case Memory::Region::device:
		{
			const int position=Memory2Type<Memory::Region::device>::type::value;
			std::get<position>(_tuple).assign(s,v);
			break; 
		}
		case Memory::Region::pinned:
		{
			const int position=Memory2Type<Memory::Region::pinned>::type::value;
			std::get<position>(_tuple).assign(s,v);
			break; 
		}
		case Memory::Region::unified:
		{
			const int position=Memory2Type<Memory::Region::unified>::type::value;
			std::get<position>(_tuple).assign(s,v);
			break; 
		}
	}
};

template<typename T>
template<typename iter> 
void column<T>::assign(iter it_1,iter it_2){
	typedef typename Traits::device_column::iterator device_it; 
	typedef typename Traits::host_column::iterator host_it;
	switch(getlocation())
	{
		case Memory::Region::host:
		{
			const int position=Memory2Type<Memory::Region::host>::type::value;
			std::get<position>(_tuple).assign(it_1,it_2);
			break; 
		}
		case Memory::Region::device:
		{
			const int position=Memory2Type<Memory::Region::device>::type::value;
			device_it ptr_1(it_1); 
			device_it ptr_2(it_2); 
			std::get<position>(_tuple).assign(ptr_1,ptr_2);
			break; 
		}
		case Memory::Region::pinned:
		{
			const int position=Memory2Type<Memory::Region::pinned>::type::value;
			device_it ptr_1(it_1); 
			device_it ptr_2(it_2); 
			std::get<position>(_tuple).assign(ptr_1,ptr_2);
			break; 
		}
		case Memory::Region::unified:
		{
			const int position=Memory2Type<Memory::Region::unified>::type::value;
			device_it ptr_1(it_1); 
			device_it ptr_2(it_2); 
			std::get<position>(_tuple).assign(ptr_1,ptr_2);
			break; 
		}
	}
};

template<typename T>
template<typename iter> 
iter column<T>::insert(iter it, column<T>::value_type v){	
	typedef typename Traits::device_column::iterator device_it; 
	typedef typename Traits::host_column::iterator host_it;
	iter out=it;
	switch(getlocation())
	{
		case Memory::Region::host:
		{
			const int position=Memory2Type<Memory::Region::host>::type::value;
			host_it ptr_h(it); 
			std::get<position>(_tuple).insert(ptr_h,v);
			break; 
		}
		case Memory::Region::device:
		{
			const int position=Memory2Type<Memory::Region::device>::type::value;
			device_it ptr_d(it); 
			std::get<position>(_tuple).insert(ptr_d,v);
			break; 
		}
		case Memory::Region::pinned:
		{
			const int position=Memory2Type<Memory::Region::pinned>::type::value;
			device_it ptr_d(it); 
			std::get<position>(_tuple).insert(ptr_d,v);
			break; 
		}
		case Memory::Region::unified:
		{
			const int position=Memory2Type<Memory::Region::unified>::type::value;
			device_it ptr_d(it); 
			std::get<position>(_tuple).insert(ptr_d,v);
			break; 
		}
	}
	return out; 
};

template<typename T>
template<typename iter_pos,typename iter> 
void column<T>::insert(iter_pos pos,iter start,iter stop){
	typedef typename Traits::host_column::iterator host_it;
	typedef typename Traits::device_column::iterator device_it; 
	switch(getlocation())
	{
		case Memory::Region::host:
		{
			const int position=Memory2Type<Memory::Region::host>::type::value;
			host_it ptr_pos(pos); 
			host_it ptr_start(start); 
			host_it ptr_stop(stop); 		
			std::get<position>(_tuple).insert(ptr_pos,ptr_start,ptr_stop);
			break; 
		}
		case Memory::Region::device:
		{
			const int position=Memory2Type<Memory::Region::device>::type::value;
			device_it ptr_pos(pos); 
			device_it ptr_start(start); 
			device_it ptr_stop(stop); 		
			std::get<position>(_tuple).insert(ptr_pos,ptr_start,ptr_stop);
			break; 
		}
		case Memory::Region::pinned:
		{
			const int position=Memory2Type<Memory::Region::pinned>::type::value;
			device_it ptr_pos(pos); 
			device_it ptr_start(start); 
			device_it ptr_stop(stop); 		
			std::get<position>(_tuple).insert(ptr_pos,ptr_start,ptr_stop);
			break; 
		}
		case Memory::Region::unified:
		{
			const int position=Memory2Type<Memory::Region::unified>::type::value;
			device_it ptr_pos(pos); 
			device_it ptr_start(start); 
			device_it ptr_stop(stop); 		
			std::get<position>(_tuple).insert(ptr_pos,ptr_start,ptr_stop);
			break; 
		}
	}
};

template<typename T>
template<typename iter> 
iter column<T>::erase(iter pos){
	typedef typename Traits::host_column::iterator host_it;
	typedef typename Traits::device_column::iterator device_it; 
	iter out; 
	switch(getlocation())
	{
		case Memory::Region::host:
		{
			const int position=Memory2Type<Memory::Region::host>::type::value;
			out=std::get<position>(_tuple).erase(pos);
			break; 
		}
		case Memory::Region::device:
		{
			const int position=Memory2Type<Memory::Region::device>::type::value;
			device_it ptr_pos(pos); 
			out=std::get<position>(_tuple).erase(ptr_pos);
			break; 
		}
		case Memory::Region::pinned:
		{
			const int position=Memory2Type<Memory::Region::pinned>::type::value;
			device_it ptr_pos(pos); 
			out=std::get<position>(_tuple).erase(ptr_pos);
			break; 
		}
		case Memory::Region::unified:
		{
			const int position=Memory2Type<Memory::Region::unified>::type::value;
			device_it ptr_pos(pos); 
			out=std::get<position>(_tuple).erase(ptr_pos);
			break; 
		}
	}
	return out; 
};

template<typename T>
template<typename iter> 
iter column<T>::erase(iter start, iter stop){
	typedef typename Traits::device_column::iterator device_it; 
	typedef typename Traits::host_column::iterator host_it;
	switch(getlocation())
	{
		case Memory::Region::host:
		{
			const int position=Memory2Type<Memory::Region::host>::type::value;
			std::get<position>(_tuple).erase(start,stop);
			break; 
		}
		case Memory::Region::device:
		{
			const int position=Memory2Type<Memory::Region::device>::type::value;
			device_it ptr_start(start); 
			device_it ptr_stop(stop); 
			std::get<position>(_tuple).erase(ptr_start,ptr_stop);
			break; 
		}
		case Memory::Region::pinned:
		{
			const int position=Memory2Type<Memory::Region::pinned>::type::value;
			device_it ptr_start(start); 
			device_it ptr_stop(stop); 
			std::get<position>(_tuple).erase(ptr_start,ptr_stop);
		break; 
		}
		case Memory::Region::unified:
		{
			const int position=Memory2Type<Memory::Region::unified>::type::value;
			device_it ptr_start(start); 
			device_it ptr_stop(stop); 
			std::get<position>(_tuple).erase(ptr_start,ptr_stop);
			break; 
		}
	}
	return start; 
};



#undef DEFAULT
