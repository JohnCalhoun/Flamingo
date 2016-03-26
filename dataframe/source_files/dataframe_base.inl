//dataframe_base.inl
#include "dataframe_base.cpp"

//--------------------base------------
dataframeBase::dataframeBase(){
	_key=_cordinator.insert(this); 
}
dataframeBase::~dataframeBase(){
	_cordinator.remove(id()); 
}
dataframeBase::Value dataframeBase::find(dataframeBase::Key key){
	return _cordinator.find(key); 
}
dataframeBase::Key dataframeBase::id(){
	return _key; 
}
void dataframeBase::id(int x){
	_cordinator.change(id(),x); 
	_key=x; 
}

