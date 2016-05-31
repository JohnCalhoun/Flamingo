template<typename T>
Internal::Cordinate<T>::size_type 
	Internal::Cordinate<T>::width()const
{
	return end-begin;
};
template<typename T>
Internal::Cordinate<T>::size_type 
	Internal::Cordinate<T>::row()const
{
	return _data.first;
};
template<typename T>
Internal::Cordinate<T>::size_type 
	Internal::Cordinate<T>::offset()const
{
	return _data.second; 
};
template<typename T>
Internal::Cordinate<T>::size_type 
	Internal::Cordinate<T>::distance()const
{
	return row()*width()+offset(); 
};
template<typename T>
void Internal::Cordinate<T>::setRow(Internal::Cordinate<T>::size_type x){
	_data.first=x;
};
template<typename T>
void Internal::Cordinate<T>::setOffset(Internal::Cordinate<T>::size_type x){
	_data.second=x;
};
template<typename T>
void Internal::Cordinate<T>::setDistance(Internal::Cordinate<T>::size_type x){
	if(width()!=0){
		setRow(x/width());
		setOffset(x%width()); 
	}else{
		setRow(0);
		setOffset(0);
	}
};
template<typename T>
template<typename A>
void Internal::Cordinate<T>::setTree(const Tree<T,A>& t){
	begin=t.cbegin(); 
	end=t.cend(); 
};
template<typename T>
void Internal::Cordinate<T>::set(Internal::Cordinate<T>::size_type x, Internal::Cordinate<T>::size_type y){
	setRow(x);
	setOffset(y);
};
template<typename T>
bool Internal::Cordinate<T>::operator<(Internal::Cordinate<T> other){
	bool result;

	if(this->begin==other.begin){
		result=this->distance()<other.distance(); 
	}else{
		result=this->begin<other.begin; 
	}

	return result;	
};
template<typename T>
bool Internal::Cordinate<T>::operator>(Internal::Cordinate<T> other){
	return other<*this;	
};
template<typename T>
Internal::Cordinate<T>::pointer Internal::Cordinate<T>::access(){
	pointer output; 
	#ifdef __CUDA_ARCH__
		pointer branch( *(begin+row()) ); 
		output=branch+offset();
	#else
		Memory::location<Memory::Region::device>::MemCopy(
								begin+row(),
								&output,
								sizeof(pointer)); 
		output+=offset(); 	
	#endif

	return output; 
};	

template<typename T>
Internal::Cordinate<T>& 
	Internal::Cordinate<T>::operator++()
{
	operator+(1); 
	return *this;
};
template<typename T>
Internal::Cordinate<T> 
	Internal::Cordinate<T>::operator++(int x)
{
	Cordinate tmp(*this); 
	operator++(); 
	return tmp; 
};
template<typename T>
Internal::Cordinate<T>& 
	Internal::Cordinate<T>::operator--()
{
	operator-(1); 
	return *this; 
};
template<typename T>
Internal::Cordinate<T> 
	Internal::Cordinate<T>::operator--(int x)
{
	Cordinate tmp(*this); 
	operator--(); 
	return tmp; 
};
template<typename T>
Internal::Cordinate<T>& 
	Internal::Cordinate<T>::operator+=(Internal::Cordinate<T>::size_type x)
{
	size_type d=distance(); 
	d+=x; 
	setDistance(d);
	return *this; 
};
template<typename T>
Internal::Cordinate<T>& 
	Internal::Cordinate<T>::operator-=(Internal::Cordinate<T>::size_type x)
{
	size_type d=distance(); 
	d-=x; 
	setDistance(d);
	return *this; 
};
template<typename T>
Internal::Cordinate<T> 
	Internal::Cordinate<T>::operator+(Internal::Cordinate<T>::size_type x)
{
	Cordinate tmp(*this);
	tmp+=x;
	return tmp; 
};
template<typename T>
Internal::Cordinate<T> 
	Internal::Cordinate<T>::operator-(Internal::Cordinate<T>::size_type x)
{
	Cordinate tmp(*this); 
	return tmp+(-x); 
};
template<typename T>
Internal::Cordinate<T>::size_type 
	Internal::Cordinate<T>::operator-( Internal::Cordinate<T> other)
{
	size_type top=other.distance(); 
	size_type bottom=other.distance();
	return top-bottom; 
};
//*****************************Cordinate*********************






