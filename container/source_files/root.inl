#define __both__ __device__ __host__
//********************************Base*******************************
template<typename T,typename A,typename L>
rootBase<T,A,L>::rootBase(){
	_size=0;
}
template<typename T,typename A,typename L>
rootBase<T,A,L>::rootBase(rootBase<T,A,L>::size_type s){
	_size=s;
}
template<typename T,typename A,typename L>
rootBase<T,A,L>::~rootBase(){}

template<typename T,typename A,typename L>
void rootBase<T,A,L>::clear(){
	if(root()){
		deallocate();
		null();
	}
	_size=0; 
}
template<typename T,typename A,typename L>
void rootBase<T,A,L>::resize(rootBase<T,A,L>::size_type s){
	if(s>0){
		allocate(s); 
		_size=s;
	}else{
		clear();
	}
}
template<typename T,typename A,typename L>
rootBase<T,A,L>::size_type rootBase<T,A,L>::size()const{
	return _size;  
}
template<typename T,typename A,typename L>
void rootBase<T,A,L>::set(	rootBase<T,A,L>::size_type index){
	Pointer p=allocatorel.allocate(this->size()); 	
	set(p,index);
}
template<typename T,typename A,typename L>
template<typename O>
void rootBase<T,A,L>::assign(const O& other){
	clear();
     if(other.size()>0){
		if(other.size()!=this->size() or !this->root() )
			this->resize( other.size() );
			int size=this->size()*sizeof(T);

			for(int x=0;x<this->size();x++){
				Pointer src=other.get(x);
				Pointer dst=this->get(x);
				if(src){
					if(!dst){
						set(x);
						dst=this->get(x);
					}
					Location_dev::MemCopy(src,dst,size);
				}else{
					break;
				}
			}
		}
}
//********************************Root Single************************
template<typename T,typename A,typename L>
rootSingle<T,A,L>::rootSingle(){
	null();
	rootBase<T,A,L>::resize(0); 
}
template<typename T,typename A,typename L>
rootSingle<T,A,L>::rootSingle(rootSingle<T,A,L>::size_type s){
	null();
	rootBase<T,A,L>::resize(s);
}
template<typename T,typename A,typename L>
rootSingle<T,A,L>::~rootSingle(){
	rootBase<T,A,L>::clear();
}
template<typename T,typename A,typename L>
rootSingle<T,A,L>& rootSingle<T,A,L>::operator=(const rootSingle<T,A,L>& other){
	rootBase<T,A,L>::assign(other); 
	return *this;
}
template<typename T,typename A,typename L>
void rootSingle<T,A,L>::allocate(	rootSingle<T,A,L>::size_type s){
	_root=allocator.allocate(s);
	for(int i=0; i<s;i++){
		_root[i]=NULL; 
	}
}

template<typename T,typename A,typename L>
void rootSingle<T,A,L>::deallocate(){
	allocator.deallocate(_root);
	null();
}

template<typename T,typename A,typename L>
void rootSingle<T,A,L>::null(){
	_root=NULL;
}

template<typename T,typename A,typename L>
bool rootSingle<T,A,L>::root(){
	bool result=_root;
	return result;
}

template<typename T,typename A,typename L>
rootSingle<T,A,L>::Pointer rootSingle<T,A,L>::get(	rootSingle<T,A,L>::size_type index)const{
	return *(_root+index);
}

template<typename T,typename A,typename L>
void rootSingle<T,A,L>::set(	const rootSingle<T,A,L>::Pointer& p,
						rootSingle<T,A,L>::size_type index){
	*(_root+index)=p;
}
//********************************Root Single************************
//********************************Root Double************************
template<typename T,typename A,typename L>
rootDouble<T,A,L>::rootDouble(){
	null();
	rootBase<T,A,L>::resize(0);
}
template<typename T,typename A,typename L>
rootDouble<T,A,L>::rootDouble(rootDouble<T,A,L>::size_type size){
	null();
	rootBase<T,A,L>::resize(size);
}
template<typename T,typename A,typename L>
rootDouble<T,A,L>::~rootDouble(){
	rootBase<T,A,L>::clear(); 
}
template<typename T,typename A,typename L>
void rootDouble<T,A,L>::sync(){
	Location_dev::MemCopy(	_rootHost,
						_rootDevice,
						rootBase<T,A,L>::size()*sizeof(Pointer)
					); 
}
template<typename T,typename A,typename L>
rootDouble<T,A,L>& rootDouble<T,A,L>::operator=(const rootDouble<T,A,L>& other){
	rootBase<T,A,L>::assign(other); 
	sync();
	return *this;
}
template<typename T,typename A,typename L>
void rootDouble<T,A,L>::allocate(	rootDouble<T,A,L>::size_type s){
	_rootHost=allocatorHost.allocate(s);
	_rootDevice=allocatorDev.allocate(s);
	for(int i=0; i<s;i++){
		_rootHost[i]=NULL; 
	}
	sync();
}

template<typename T,typename A,typename L>
void rootDouble<T,A,L>::deallocate(){
	allocatorHost.deallocate(_rootHost);
	allocatorDev.deallocate(_rootDevice);
	null();
}

template<typename T,typename A,typename L>
void rootDouble<T,A,L>::null(){
	_rootHost=NULL;
	_rootDevice=NULL;
}

template<typename T,typename A,typename L>
bool rootDouble<T,A,L>::root(){
	bool h=_rootHost;
	bool d=_rootDevice;
	bool result=h and d;

	return result; 
}

template<typename T,typename A,typename L>
rootDouble<T,A,L>::Pointer rootDouble<T,A,L>::get(	rootDouble<T,A,L>::size_type index)const{
	#ifdef __CUDA_ARCH__
		return *(_rootDevice+index);
	#else
		return *(_rootHost+index);
	#endif
}

template<typename T,typename A,typename L>
void rootDouble<T,A,L>::set(	const rootDouble<T,A,L>::Pointer& p,
						rootDouble<T,A,L>::size_type index){
	*(_rootHost+index)=p;
	sync();
}
//********************************Root Double************************
#undef __both__
