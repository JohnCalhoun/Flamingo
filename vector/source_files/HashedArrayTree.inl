//HashedArrayTree.inl
#include<thread>
#include"HashedArrayTree.cu"
#include<cmath>
#include <functional>
#include<algorithm>
#include<location.cu>
#include<type_traits>
#include<cstdlib>
#include<utility>
//***************************CONSTRUCTORS/DESTRUCTORS********************************************
template<typename T,Memory::Region M>
HashedArrayTree<T,M>::HashedArrayTree()
	:_cap(0),
	_count(0)
{
	resize(_cap); 	
};

template<typename T,Memory::Region M>
HashedArrayTree<T,M>::HashedArrayTree(const HashedArrayTree<T,M>& other)
{
	_count=other._count; 
	_cap=other._cap; 
	_tree=other._tree; 
};
template<typename T,Memory::Region M>
HashedArrayTree<T,M>::HashedArrayTree(
		HashedArrayTree<T,M>::size_type x):HashedArrayTree(x,T())
{};

template<typename T,Memory::Region M>
HashedArrayTree<T,M>::HashedArrayTree(
		HashedArrayTree<T,M>::size_type x,T item)
		:HashedArrayTree()
{
	insert(this->begin(),x,item);
};

template<typename T,Memory::Region M>
HashedArrayTree<T,M>::~HashedArrayTree(){};

//**********CONSTRUCTORS/DESTRUCTORS***********************
//**********INTERNAL FUNCTIONS*****************************
template<typename T,Memory::Region M>
HashedArrayTree<T,M>::size_type HashedArrayTree<T,M>::calculate_width(HashedArrayTree<T,M>::size_type x){
	size_type tmp; 
	if(x==0){
		tmp=1;
	}else{
		tmp=std::ceil(	std::log2(x)/2	);
	}

	return std::pow(2,tmp); 
}

template<typename T,Memory::Region M>
void HashedArrayTree<T,M>::resize(HashedArrayTree<T,M>::size_type x){
	if(x!=size() ){
	if(x>size() ){
		size_type width_new=calculate_width(x);
		_tree.resize(width_new); 
	
		int needed_leaves	=x/_tree.width(); 	
		int current_leaves	=_tree.openbranch();

		if(needed_leaves>=current_leaves){
			int leavestoadd=needed_leaves-current_leaves+1;
			for(int i=0; i<leavestoadd;i++){
				_tree.addbranch(); 
			}
		}	
	}else{

		int needed_leaves	=x/_tree.width(); 	
		int current_leaves	=_tree.openbranch();

		if(needed_leaves<=current_leaves){
			int leavestoremove=current_leaves-needed_leaves;
			for(int i=0; i<leavestoremove;i++){
				_tree.removebranch(); 
			}
		}

		size_type width_new=calculate_width(x);
		_tree.resize(width_new); 
	}
	}
	_cap=(_tree.openbranch()-1)*_tree.width();
	_count=x; 
}

template<typename T,Memory::Region M>
HashedArrayTree<T,M>::iterator HashedArrayTree<T,M>::add_end(
			HashedArrayTree<T,M>::size_type	x,
			HashedArrayTree<T,M>::iterator	it	){
	int needed_size=size()+x;
	resize(needed_size); 

	iterator out(_tree,it);
	return out;
}

template<typename T,Memory::Region M>
void HashedArrayTree<T,M>::remove_end(HashedArrayTree<T,M>::size_type x){
	int needed_size=size()-x;
	resize(needed_size); 	
};

template<typename T,Memory::Region M>
template<typename D>
void HashedArrayTree<T,M>::shift(
		HashedArrayTree<T,M>::iterator it,
		HashedArrayTree<T,M>::size_type n)
{
	typedef std::tuple<Cordinate,Cordinate,int>	paramater;
	typedef std::vector<paramater>			paramater_vector;
	typedef Internal::shift_functions<D,paramater_vector,T,allocator_type>
			shifter; 
	int width=_tree.width(); 
	shifter util(n);

	paramater_vector param_v;
	
	Cordinate src=it._cordinate;
	Cordinate dst=src;

	while( src.distance() < capacity()){
		dst=util.next(src); 	
	
		if(dst.distance()<capacity() ){
			int copy_size=dst-src; 
			param_v.insert(	param_v.begin(),
							std::make_tuple(	src,
											dst,
											copy_size)
						); 
		}
		src=dst+1; 
	}
	util.adjust(param_v); 
	std::for_each(param_v.begin(),param_v.end(),
		[this](paramater p){
			typedef typename tree::pointer pointer; 
			Cordinate source=		std::get<0>(p);
			Cordinate destination=	std::get<1>(p);
			int s=				std::get<2>(p); 

			pointer src_it=source.access(); 
			pointer dst_it=destination.access(); 

			this->location.MemCopy(	src_it,
								dst_it,
								s*sizeof(T) ); 
		}
	);

}

template<typename T,Memory::Region M>
template<typename I>
void HashedArrayTree<T,M>::copy(I it,HashedArrayTree<T,M>::size_type n,const T& item){
	location.fill_in(it,n,item);
}
template<typename T,Memory::Region M>
template<typename I>
void HashedArrayTree<T,M>::copy(I it_in,HashedArrayTree<T,M>::size_type n,I it_out){
	location.MemCopy(it_in,it_out,n); 
}

//***********************************INTERNAL FUNCTIONS********************************************
//***********************************ITERATOR GETTERS********************************************

template<typename T,Memory::Region M>
HashedArrayTree<T,M>::iterator HashedArrayTree<T,M>::begin(){
	iterator it(_tree);
	it.initalize(0,0); 
	return it; 
};

template<typename T,Memory::Region M>
HashedArrayTree<T,M>::iterator HashedArrayTree<T,M>::end(){
	iterator it(_tree);
	it.initalize( capacity() );
	return it; 
};

template<typename T,Memory::Region M>
HashedArrayTree<T,M>::const_iterator HashedArrayTree<T,M>::cbegin()const{
	const_iterator it(_tree);
	it.initalize(0,0); 
	return it;
};

template<typename T,Memory::Region M>
HashedArrayTree<T,M>::const_iterator HashedArrayTree<T,M>::cend()const{
	const_iterator it(_tree);
	it.initalize( capacity() );
	return it; 
};


template<typename T,Memory::Region M>
HashedArrayTree<T,M>::reverse_iterator HashedArrayTree<T,M>::rbegin(){
	reverse_iterator it(_tree);
	it.initalize(0,0); 
	return it; 
};

template<typename T,Memory::Region M>
HashedArrayTree<T,M>::reverse_iterator HashedArrayTree<T,M>::rend(){	
	reverse_iterator it(_tree);
	it.initalize( capacity() );
	return it; 
};

template<typename T,Memory::Region M>
HashedArrayTree<T,M>::const_reverse_iterator HashedArrayTree<T,M>::crbegin()const{
	const_reverse_iterator it(_tree);
	it.initalize(0,0); 
	return it;
};

template<typename T,Memory::Region M>
HashedArrayTree<T,M>::const_reverse_iterator HashedArrayTree<T,M>::crend()const{
	const_reverse_iterator it(_tree);
	it.initalize( capacity() ); 
	return it; 
};

//***********************************ITERATOR GETTERS********************************************
//***********************************OPERATORS********************************************
//deal with T!=T=>false and L!=L=>false,maybe
template<typename T,Memory::Region M>
template<Memory::Region A>
HashedArrayTree<T,M>& HashedArrayTree<T,M>::operator=(const HashedArrayTree<T,A>& other){
	_cap=other.capacity();
	_count=other._count; 
	_tree=other._tree;
	return *this;
};

template<typename T,Memory::Region M>
bool HashedArrayTree<T,M>::operator==(const HashedArrayTree<T,M>& other)const{
	if(size()==other.size()){
		for(int i=0; i<size();i++){
			return  (this->at(i) ) == other.at(i); 
		}
		return (this->_tree)==(other._tree);
	}else{
		return false;
	}
};

template<typename T,Memory::Region M>
bool HashedArrayTree<T,M>::operator!=(const HashedArrayTree<T,M>& other)const{
	return !(*this==other);
};

//***********************************OPERTORS********************************************
///***********************************INTERFACE********************************************/

template<typename T,Memory::Region M>
void	HashedArrayTree<T,M>::swap(HashedArrayTree<T,M>& other){
	_tree.swap( (other._tree) );

	std::swap(this->_cap,other._cap); 
	std::swap(this->_count,other._count); 
};

template<typename T,Memory::Region M>
HashedArrayTree<T,M>::size_type HashedArrayTree<T,M>::size()const{
	return _count; 
};

template<typename T,Memory::Region M>
HashedArrayTree<T,M>::size_type HashedArrayTree<T,M>::capacity()const{
	return _cap; 
};

template<typename T,Memory::Region M>
HashedArrayTree<T,M>::size_type HashedArrayTree<T,M>::max_size()const{
	return _allocator.max_size(); 
};

template<typename T,Memory::Region M>
bool	HashedArrayTree<T,M>::empty()const{
	return size()>0;
};

template<typename T,Memory::Region M>
void HashedArrayTree<T,M>::reserve(HashedArrayTree<T,M>::size_type x){
	resize(x);
};

template<typename T,Memory::Region M>
void	HashedArrayTree<T,M>::shrink_to_fit(){
	resize( size() ); 
};

/*	
emplate<typename T,Memory::Region M>
template<class ...Args>
	HashedArrayTree<T,M>::iterator HashedArrayTree<T,M>::emplace(HashedArrayTree<T,M>::const_iterator iter, Args...){

	};
*/
template<typename T,Memory::Region M>
HashedArrayTree<T,M>::iterator HashedArrayTree<T,M>::insert(HashedArrayTree<T,M>::const_iterator iter, const T& item){
	iterator new_iter=add_end(1,iter);

	shift<DOWN>(new_iter,1);	
	copy(new_iter,1,item);
	return new_iter+1;
}

template<typename T,Memory::Region M>
HashedArrayTree<T,M>::iterator HashedArrayTree<T,M>::insert(HashedArrayTree<T,M>::const_iterator iter, T&& item){ 
	iterator new_iter=add_end(1,iter);

	shift<DOWN>(new_iter,1); 
	copy(new_iter,1,item); 
	return new_iter+1;
};

template<typename T,Memory::Region M>
HashedArrayTree<T,M>::iterator HashedArrayTree<T,M>::insert(HashedArrayTree<T,M>::const_iterator iter, HashedArrayTree<T,M>::size_type n, T& item){
	iterator new_iter=add_end(n,iter); 

	shift<DOWN>(new_iter,n); 
	copy(new_iter,n,item);
	return new_iter+1;
}
	
template<typename T,Memory::Region M>
template<class iter>
	HashedArrayTree<T,M>::iterator HashedArrayTree<T,M>::insert(HashedArrayTree<T,M>::const_iterator iter_1, iter iter_2,  iter iter_3){
 	int n=iter_2-iter_1;
	iterator new_iter_1=add_end(n,iter_1);

	shift<DOWN>(new_iter_1,n);

	copy(new_iter_1,n,iter_3);
	return new_iter_1+1;	
};
/*
specialize for location

template<typename T,Memory::Region M>
HashedArrayTree<T,M>::iterator HashedArrayTree<T,M>::insert(HashedArrayTree<T,M>::const_iterator iter, std::initializer_list<T> list){

};
*/

template<typename T,Memory::Region M>
HashedArrayTree<T,M>::iterator HashedArrayTree<T,M>::erase(HashedArrayTree<T,M>::const_iterator iter){
	shift<UP>(iter,1); 
	remove_end(1);
	return iter-1; 
};

template<typename T,Memory::Region M>
HashedArrayTree<T,M>::iterator HashedArrayTree<T,M>::erase(HashedArrayTree<T,M>::const_iterator iter_begin , HashedArrayTree<T,M>::const_iterator iter_end){
	int n=iter_end-iter_begin; 
	shift<UP>(iter_end,n);
	remove_end(n); 
	return iter_begin; 
};


template<typename T,Memory::Region M>
void	HashedArrayTree<T,M>::clear(){
	_tree.clear(); 
	_cap=0; 
	_count=0; 
};

template<typename T,Memory::Region M>
template<class iter> 
void HashedArrayTree<T,M>::assign(iter it_1, iter it_2){
	clear(); 
	size_type count=it_2-it_1; 
	resize(count); 
	for(int i=0; i<count; i++){
		at(i)=*(it_1+i); 
	}
};

template<typename T,Memory::Region M>
void	HashedArrayTree<T,M>::assign(std::initializer_list<T> list){
	assign(list.begin(),list.end()); 
};

template<typename T,Memory::Region M>
void	HashedArrayTree<T,M>::assign(HashedArrayTree<T,M>::size_type n, const T& item){
	clear();
	resize(n);
	copy(this->begin(),n,item); 
};

template<typename T,Memory::Region M>
HashedArrayTree<T,M>::allocator_type HashedArrayTree<T,M>::get_allocator(){
	return _allocator; 
};

template<typename T,Memory::Region M>
HashedArrayTree<T,M>::reference HashedArrayTree<T,M>::operator[](HashedArrayTree<T,M>::size_type x){
	return at(x);
};
template<typename T,Memory::Region M>
HashedArrayTree<T,M>::reference HashedArrayTree<T,M>::at(HashedArrayTree<T,M>::size_type x){
	iterator tmp=begin()+x; 
	return *tmp;
};
template<typename T,Memory::Region M>
HashedArrayTree<T,M>::reference HashedArrayTree<T,M>::front(){
	return at(0);
};
template<typename T,Memory::Region M>
HashedArrayTree<T,M>::reference HashedArrayTree<T,M>::back(){
	return at(size() ); 
};

template<typename T,Memory::Region M>
HashedArrayTree<T,M>::const_reference HashedArrayTree<T,M>::operator[](HashedArrayTree<T,M>::size_type x)const{
	return at(x);
};
template<typename T,Memory::Region M>
HashedArrayTree<T,M>::const_reference HashedArrayTree<T,M>::at(HashedArrayTree<T,M>::size_type x)const{
	iterator tmp=cbegin()+x; 
	return *tmp;
};
template<typename T,Memory::Region M>
HashedArrayTree<T,M>::const_reference HashedArrayTree<T,M>::front()const{
	return at(0);
};
template<typename T,Memory::Region M>
HashedArrayTree<T,M>::const_reference HashedArrayTree<T,M>::back()const{
	return at(size() ); 
};









