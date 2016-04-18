//cordinator.inl
#ifndef CORDINATOR_INL_CPP
#define CORDINATOR_INL_CPP

#include "cordinator.cpp"

template<typename Object,typename Guard>
cordinator<Object,Guard>::~cordinator(){}

template<typename Object,typename Guard>
cordinator<Object,Guard>::Key cordinator<Object,Guard>::insert(Object* ob){
	Key key=reinterpret_cast<Key>(ob);	
	key=insert(key,ob); 
	return key; 
} 
template<typename Object,typename Guard>
cordinator<Object,Guard>::Key cordinator<Object,Guard>::insert(
				cordinator<Object,Guard>::Key key,
				cordinator<Object,Guard>::Value value)
			
{
	r_accessor access;
	bool result; 

	result=_map.insert(access,value_type(key,value) );

	if(!result){
		//maybe throw exception, need way to enfore uniquness of ids
		result=_map.insert(access,value_type(key++,value));
	}
	return key; 
} 

template<typename Object,typename Guard>
void cordinator<Object,Guard>::remove(cordinator<Object,Guard>::Key key){
	_map.erase(key); 
}

template<typename Object,typename Guard>
cordinator<Object,Guard>::Value cordinator<Object,Guard>::find(
					cordinator<Object,Guard>::Key key)
{
	w_accessor access;
	_map.find(access,key);
	return std::get<1>(*(access));
}

template<typename Object,typename Guard>
void cordinator<Object,Guard>::change(
					cordinator<Object,Guard>::Key Old,
					cordinator<Object,Guard>::Key New)
{
	Value value=find(Old); 
	remove(Old); 
	insert(New,value);  
}
template<typename Object,typename Guard>
void cordinator<Object,Guard>::move(
					cordinator<Object,Guard>::Key key,
					Memory::Region M,
					cordinator<Object,Guard>::lock_guard& guard)
{
	Memory::Region current=find(key)->location(); 
	if(M != current){
		bool was_released=guard->upgrade_to_writer(); 
		current=find(key)->location();	
		if(was_released || (M != current)){ 
			_cache.request(key,M); 
		}//end if
	}//end if
}//end move

#endif











