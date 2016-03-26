//addressbook.inl
#include "addressbook.cpp"

template<typename Object>
addressbook<Object>::addressbook(){}

template<typename Object>
addressbook<Object>::~addressbook(){}

template<typename Object>
addressbook<Object>::Key addressbook<Object>::insert(Object* ob){
	Key key=reinterpret_cast<Key>(ob);	
	key=insert(key,ob); 
	return key; 
} 
template<typename Object>
addressbook<Object>::Key addressbook<Object>::insert(
				addressbook<Object>::Key key,
				addressbook<Object>::Value value)
			
{
	r_accessor access;
	bool result; 

	result=_map.insert(access,value_type(key,value) );

	while(!result){
		result=_map.insert(access,value_type(key++,value));
	}
	return key; 
} 

template<typename Object>
void addressbook<Object>::remove(addressbook<Object>::Key key){
	_map.erase(key); 
}

template<typename Object>
addressbook<Object>::Value addressbook<Object>::find(
					addressbook<Object>::Key key)
{
	w_accessor access;
	_map.find(access,key);
	return std::get<1>(*(access));
}

template<typename Object>
void addressbook<Object>::change(
					addressbook<Object>::Key Old,
					addressbook<Object>::Key New)
{
	Value value=find(Old); 
	remove(Old); 
	insert(New,value);  
}

template<typename Object>
addressbook<Object>::iterator addressbook<Object>::begin()
{
	return _map.begin(); 
}
template<typename Object>
addressbook<Object>::iterator addressbook<Object>::end()
{
	return _map.end(); 
}

















