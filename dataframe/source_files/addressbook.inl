//addressbook.inl
#include "addressbook.cpp"

template<typename Object>
addressbook<Object>::addressbook(){}

template<typename Object>
addressbook<Object>::~addressbook(){}

template<typename Object>
addressbook<Object>::Key addressbook<Object>::objectToKey(Object* object){
	Key key=reinterpret_cast<Key>(object);
	int count=_map.count(key);
	while(count!=0){
		key++;
		count=_map.count(key);
	}
	return key;
}

template<typename Object>
addressbook<Object>::Key addressbook<Object>::insert(Object* ob){
	lock_guard guard(_mutex);

	Key key=objectToKey(ob); 
	Value value=ob;
 
	_map.insert(std::make_pair(key,value));
	return key; 
} 
template<typename Object>
void addressbook<Object>::insert(
				addressbook<Object>::Key key,
				addressbook<Object>::Value value)
			
{
	lock_guard guard(_mutex);
	_map.insert({key,value});
} 

template<typename Object>
void addressbook<Object>::remove(addressbook<Object>::Key key){
	lock_guard guard(_mutex);

	iterator it=_map.find(key);
	_map.erase(it); 
}

template<typename Object>
addressbook<Object>::Value addressbook<Object>::find(
					addressbook<Object>::Key key)
{
	shared_guard guard(_mutex); 
	return std::get<1>(*(_map.find(key)));
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


















