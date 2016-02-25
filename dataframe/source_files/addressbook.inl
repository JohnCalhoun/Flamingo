//addressbook.inl
#include "addressbook.cpp"

template<typename Object>
addressbook<Object>::addressbook(){}

template<typename Object>
addressbook<Object>::~addressbook(){
//could provide garbage collection utilites here 
}

template<typename Object>
addressbook<Object>::Key addressbook<Object>::objectToKey(const Object& object)const{
	Key key=static_cast<Key>(&object);
	return key;
}

template<typename Object>
template<typename T>
void addressbook<Object>::insert(const T& t){
	lock_guard guard(_mutex);

	T* child=&t;
	Value value=dynamic_cast<Value>(child);
	_map.insert(value);
} 

template<typename Object>
void addressbook<Object>::remove(addressbook<Object>::Key key){
	lock_guard guard(_mutex);

	iterator it=_map.find(key);
	_map.erase(it); 
}

template<typename Object>
template<typename T>
T* addressbook<Object>::find(addressbook<Object>::Key key)const{
	shared_guard guard(_mutex); 

	iterator it=_map.find(key);
	T* out=dynamic_cast<T*>(it); 
	return out; 
}

