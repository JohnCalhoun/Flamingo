//columns.inl
#include"columns.cpp"
#include <tuple>
#define DEFAULT_LOCATION host
#define DEFAULT_COLUMN host_column

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
	const int destination=memory2type<device>::type::value;
	std::get<destination>(_tuple)=other; 
	_location=device;
} 

template<typename T>
template<typename Aloc>
column<T>::column(const thrust::host_vector<T,Aloc>& other){
	const int destination=memory2type<host>::type::value;
	std::get<destination>(_tuple)=other; 
	_location=host;
} 

template<typename T>
template<Memory M>
void column<T>::move(){
	if( M != _location){
		switch(getlocation())
		{
			case host:
			{
				const int source=memory2type<host>::type::value;
				const int destination=memory2type<M>::type::value;

				std::get<destination>(_tuple)=std::get<source>(_tuple); 
				std::get<source>(_tuple).clear(); 
				break;
			}
			case device:
			{
				const int source=memory2type<device>::type::value;
				const int destination=memory2type<M>::type::value;

				std::get<destination>(_tuple)=std::get<source>(_tuple); 
				std::get<source>(_tuple).clear(); 
				break;
			}
			case pinned:
			{
				const int source=memory2type<pinned>::type::value;
				const int destination=memory2type<M>::type::value;

				std::get<destination>(_tuple)=std::get<source>(_tuple); 
				std::get<source>(_tuple).clear(); 
				break;
			}
			case unified:
			{
				const int source=memory2type<unified>::type::value;
				const int destination=memory2type<M>::type::value;

				std::get<destination>(_tuple)=std::get<source>(_tuple); 
				std::get<source>(_tuple).clear(); 
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
typename column<T>::Return<M>::column& column<T>::access(){
	const int position=memory2type<M>::type::value;
	return std::get<position>(_tuple); 
} 
template<typename T>
column<T>::pointer column<T>::access_raw(){
	pointer ptr;	
	switch(getlocation())
	{
			case host:
			{
				const int position=memory2type<host>::type::value;
				ptr=std::get<position>(_tuple).data();
				break;
			}
			case device:
			{
				const int position=memory2type<device>::type::value;
				ptr=std::get<position>(_tuple).data();
				break;
			}
			case pinned:
			{
				const int position=memory2type<pinned>::type::value;
				ptr=std::get<position>(_tuple).data();
				break;
			}
			case unified:
			{
				const int position=memory2type<unified>::type::value;
				ptr=std::get<position>(_tuple).data();
				break;
			}
	}
	return ptr; 
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
column<T>::size_type column<T>::size()const
{
	size_type size; 
	switch(getlocation())
	{
		case host:
		{
			const int position=memory2type<host>::type::value;
			size=std::get<position>(_tuple).size();
			break; 
		}
		case device:
		{
			const int position=memory2type<device>::type::value;
			size=std::get<position>(_tuple).size();
			break; 
		}
		case pinned:
		{
			const int position=memory2type<pinned>::type::value;
			size=std::get<position>(_tuple).size();
			break; 
		}
		case unified:
		{
			const int position=memory2type<unified>::type::value;
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
		case host:
		{
			const int position=memory2type<host>::type::value;
			max_size=std::get<position>(_tuple).max_size();
			break; 
		}
		case device:
		{
			const int position=memory2type<device>::type::value;
			max_size=std::get<position>(_tuple).max_size();
			break; 
		}
		case pinned:
		{
			const int position=memory2type<pinned>::type::value;
			max_size=std::get<position>(_tuple).max_size();
			break; 
		}
		case unified:
		{
			const int position=memory2type<unified>::type::value;
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
		case host:
		{
			const int position=memory2type<host>::type::value;
			empty=std::get<position>(_tuple).empty();
			break; 
		}
		case device:
		{
			const int position=memory2type<device>::type::value;
			empty=std::get<position>(_tuple).empty();
			break; 
		}
		case pinned:
		{
			const int position=memory2type<pinned>::type::value;
			empty=std::get<position>(_tuple).empty();
			break; 
		}
		case unified:
		{
			const int position=memory2type<unified>::type::value;
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
		case host:
		{
			const int position=memory2type<host>::type::value;
			std::get<position>(_tuple).reserve(size);
			break; 
		}
		case device:
		{
			const int position=memory2type<device>::type::value;
			std::get<position>(_tuple).reserve(size);
			break; 
		}
		case pinned:
		{
			const int position=memory2type<pinned>::type::value;
			std::get<position>(_tuple).reserve(size);
			break; 
		}
		case unified:
		{
			const int position=memory2type<unified>::type::value;
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
		case host:
		{
			const int position=memory2type<host>::type::value;
			size=std::get<position>(_tuple).capacity();
			break; 
		}
		case device:
		{
			const int position=memory2type<device>::type::value;
			size=std::get<position>(_tuple).capacity();
			break; 
		}
		case pinned:
		{
			const int position=memory2type<pinned>::type::value;
			size=std::get<position>(_tuple).capacity();
			break; 
		}
		case unified:
		{
			const int position=memory2type<unified>::type::value;
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
		case host:
		{
			const int position=memory2type<host>::type::value;
			std::get<position>(_tuple).assign(s,t);
			break; 
		}
		case device:
		{
			const int position=memory2type<device>::type::value;
			std::get<position>(_tuple).assign(s,t);
			break; 
		}
		case pinned:
		{
			const int position=memory2type<pinned>::type::value;
			std::get<position>(_tuple).assign(s,t);
			break; 
		}
		case unified:
		{
			const int position=memory2type<unified>::type::value;
			std::get<position>(_tuple).assign(s,t);
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
			const int position=memory2type<host>::type::value;
			std::get<position>(_tuple).assign(start,stop);
			break; 
		}
		case device:
		{
			const int position=memory2type<device>::type::value;
			std::get<position>(_tuple).assign(start,stop);
			break; 
		}
		case pinned:
		{
			const int position=memory2type<pinned>::type::value;
			std::get<position>(_tuple).assign(start,stop);
			break; 
		}
		case unified:
		{
			const int position=memory2type<unified>::type::value;
			std::get<position>(_tuple).assign(start,stop);
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
			const int position=memory2type<host>::type::value;
			std::get<position>(_tuple).clear();
			break; 
		}
		case device:
		{
			const int position=memory2type<device>::type::value;
			std::get<position>(_tuple).clear();
			break; 
		}
		case pinned:
		{
			const int position=memory2type<pinned>::type::value;
			std::get<position>(_tuple).clear();
			break; 
		}
		case unified:
		{
			const int position=memory2type<unified>::type::value;
			std::get<position>(_tuple).clear();
			break; 
		}
	}
}

template<typename T>
void column<T>::resize(column<T>::size_type n){
	switch(getlocation())
	{
		case host:
		{
			const int position=memory2type<host>::type::value;
			std::get<position>(_tuple).resize(n);
			break; 
		}
		case device:
		{
			const int position=memory2type<device>::type::value;
			std::get<position>(_tuple).resize(n);
			break; 
		}
		case pinned:
		{
			const int position=memory2type<pinned>::type::value;
			std::get<position>(_tuple).resize(n);
			break; 
		}
		case unified:
		{
			const int position=memory2type<unified>::type::value;
			std::get<position>(_tuple).resize(n);
			break; 
		}
	}
}

template<typename T>
void column<T>::resize(column<T>::size_type n,column<T>::value_type v){
	switch(getlocation())
	{
		case host:
		{
			const int position=memory2type<host>::type::value;
			std::get<position>(_tuple).resize(n,v);
			break; 
		}
		case device:
		{
			const int position=memory2type<device>::type::value;
			std::get<position>(_tuple).resize(n,v);
			break; 
		}
		case pinned:
		{
			const int position=memory2type<pinned>::type::value;
			std::get<position>(_tuple).resize(n,v);
			break; 
		}
		case unified:
		{
			const int position=memory2type<unified>::type::value;
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
		case host:
		{
			const int position=memory2type<host>::type::value;
			std::get<position>(_tuple).assign(s,v);
			break; 
		}
		case device:
		{
			const int position=memory2type<device>::type::value;
			std::get<position>(_tuple).assign(s,v);
			break; 
		}
		case pinned:
		{
			const int position=memory2type<pinned>::type::value;
			std::get<position>(_tuple).assign(s,v);
			break; 
		}
		case unified:
		{
			const int position=memory2type<unified>::type::value;
			std::get<position>(_tuple).assign(s,v);
			break; 
		}
	}
};

template<typename T>
template<typename iter> 
void column<T>::assign(iter it_1,iter it_2){
	switch(getlocation())
	{
		case host:
		{
			const int position=memory2type<host>::type::value;
			std::get<position>(_tuple).assign(it_1,it_2);
			break; 
		}
		case device:
		{
			const int position=memory2type<device>::type::value;
			std::get<position>(_tuple).assign(it_1,it_2);
			break; 
		}
		case pinned:
		{
			const int position=memory2type<pinned>::type::value;
			std::get<position>(_tuple).assign(it_1,it_2);
			break; 
		}
		case unified:
		{
			const int position=memory2type<unified>::type::value;
			std::get<position>(_tuple).assign(it_1,it_2);
			break; 
		}
	}
};

template<typename T>
template<typename iter> 
iter column<T>::insert(iter it, column<T>::value_type v){	
	switch(getlocation())
	{
		case host:
		{
			const int position=memory2type<host>::type::value;
			std::get<position>(_tuple).insert(it,v);
			break; 
		}
		case device:
		{
			const int position=memory2type<device>::type::value;
			std::get<position>(_tuple).insert(it,v);
			break; 
		}
		case pinned:
		{
			const int position=memory2type<pinned>::type::value;
			std::get<position>(_tuple).insert(it,v);
			break; 
		}
		case unified:
		{
			const int position=memory2type<unified>::type::value;
			std::get<position>(_tuple).insert(it,v);
			break; 
		}
	}
	return it; 
};

template<typename T>
template<typename iter_pos,typename iter> 
void column<T>::insert(iter_pos pos,iter start,iter stop){
	switch(getlocation())
	{
		case host:
		{
			const int position=memory2type<host>::type::value;
			std::get<position>(_tuple).insert(pos,start,stop);
			break; 
		}
		case device:
		{
			const int position=memory2type<device>::type::value;
			std::get<position>(_tuple).insert(pos,start,stop);
			break; 
		}
		case pinned:
		{
			const int position=memory2type<pinned>::type::value;
			std::get<position>(_tuple).insert(pos,start,stop);
			break; 
		}
		case unified:
		{
			const int position=memory2type<unified>::type::value;
			std::get<position>(_tuple).insert(pos,start,stop);
			break; 
		}
	}
};

template<typename T>
template<typename iter> 
iter column<T>::erase(iter pos){
	switch(getlocation())
	{
		case host:
		{
			const int position=memory2type<host>::type::value;
			std::get<position>(_tuple).erase(pos);
			break; 
		}
		case device:
		{
			const int position=memory2type<device>::type::value;
			std::get<position>(_tuple).erase(pos);
			break; 
		}
		case pinned:
		{
			const int position=memory2type<pinned>::type::value;
			std::get<position>(_tuple).erase(pos);
			break; 
		}
		case unified:
		{
			const int position=memory2type<unified>::type::value;
			std::get<position>(_tuple).erase(pos);
			break; 
		}
	}
	return pos; 
};

template<typename T>
template<typename iter> 
iter column<T>::erase(iter start, iter stop){
	switch(getlocation())
	{
		case host:
		{
			const int position=memory2type<host>::type::value;
			std::get<position>(_tuple).erase(start,stop);
			break; 
		}
		case device:
		{
			const int position=memory2type<device>::type::value;
			std::get<position>(_tuple).erase(start,stop);
			break; 
		}
		case pinned:
		{
			const int position=memory2type<pinned>::type::value;
			std::get<position>(_tuple).erase(start,stop);
			break; 
		}
		case unified:
		{
			const int position=memory2type<unified>::type::value;
			std::get<position>(_tuple).erase(start,stop);
			break; 
		}
	}
	return start; 
};



#undef DEFAULT
