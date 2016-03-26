//dataframe_base.inl
#include "dataframe_base.cpp"

//--------------------base------------
dataframeBase::dataframeBase(){
	_key=_addressbook.insert(this); 
}
dataframeBase::~dataframeBase(){
	_addressbook.remove(id()); 
}
dataframeBase::Value dataframeBase::find(dataframeBase::Key key){
	return _addressbook.find(key); 
}
dataframeBase::Key dataframeBase::id(){
	return _key; 
}
void dataframeBase::id(int x){
	_addressbook.change(id(),x); 
	_key=x; 
}

