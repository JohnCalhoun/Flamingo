//dataframe_base.inl
#include "dataframe_base.cpp"

//--------------------base------------
dataframeBase::dataframeBase(){
	_key=_cordinator.insert(this); 
	_mutex_ptr=new Mutex; 
}

dataframeBase::dataframeBase(const dataframeBase& other){
	_key=_cordinator.insert(this); 
	_mutex_ptr=new Mutex; 
}

dataframeBase::~dataframeBase(){
	_cordinator.remove(id()); 
	delete _mutex_ptr; 
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

dataframeBase::lock_guard dataframeBase::use(Memory::Region M)
{
	bool write=false;
	lock_guard guard=lock(write);
	_cordinator.move(id(),M,guard); 
	return guard; 
};

std::tuple<	dataframeBase::lock_guard,
			bool> 
	dataframeBase::try_lock(bool write)
{
	scoped_lock* guard=new scoped_lock();
	bool result=guard->try_acquire(*_mutex_ptr,write);
	if(!result){
		guard=NULL; 
	}
	lock_guard return_guard(guard); 
	return std::make_tuple(return_guard,result); 
};

dataframeBase::lock_guard dataframeBase::lock(bool write)
{
	scoped_lock* guard=new scoped_lock(*_mutex_ptr,write);
	return lock_guard(guard); 
};

void dataframeBase::release(dataframeBase::lock_guard& lock_ptr){
	lock_ptr=NULL; 
};
