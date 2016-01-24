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
template<typename T,typename L>
HashedArrayTree<T,L>::HashedArrayTree(){
	_size=0;
	resize(_size); 	
};

template<typename T,typename L>
HashedArrayTree<T,L>::HashedArrayTree(const HashedArrayTree<T,L>& other){
	_size=other._size; 
	_tree=other._tree; 
};
template<typename T,typename L>
HashedArrayTree<T,L>::HashedArrayTree(int x,T item){
	resize(x);
	insert(this->begin(),x,item);
};

template<typename T,typename L>
HashedArrayTree<T,L>::~HashedArrayTree(){};

//***********************************CONSTRUCTORS/DESTRUCTORS********************************************
//***********************************INTERNAL FUNCTIONS********************************************
template<typename T,typename L>
int HashedArrayTree<T,L>::calculate_width(int x){
	return std::pow(2,std::ceil( std::log2( std::sqrt(x) ) ) );
}

template<typename T,typename L>
void HashedArrayTree<T,L>::resize(int x){
	if(x>size() ){
		if(x>capacity() ){
			int width_new=calculate_width(x);
			int width_current=_tree.width();
			tree tmp(width_new); 
			if(width_current!=0){
				int factor=width_new/width_current;
				for(int i=0; i<_tree.openbranch(); i++){
					std::div_t division=div(i,factor); 
					int mod=division.rem;
					int row_into=division.quot; 
					if(mod==0){
						tmp.addbranch(); 
					}
					location.MemCopy(	_tree.getbranch(i),
									tmp.getbranch(row_into)+width_current*mod,
									width_current);
				}
			}else{
				int branchs=x/width_new+1;
				for(int i=0;i<branchs;i++){
					tmp.addbranch(); 
				}
			}
			_tree=tmp; 
		}else{
			Cordinate cor;
			cor.setTree(&_tree);
			cor.setDistance(x);

			int needed_leaves=cor.row(); 
			int current_leaves=_tree.openbranch();
			if(needed_leaves>=current_leaves){
				int leavestoadd=needed_leaves-current_leaves+1;
				for(int i=0; i<leavestoadd;i++){
					_tree.addbranch(); 
				}
			}
		}
		_size=x;
	}else{
		//make smaller 
		//ignore for now
	}
}

template<typename T,typename L>
void HashedArrayTree<T,L>::add_end(int x){
	int needed_size=_size+x;
	resize(needed_size); 
}

template<typename T,typename L>
void HashedArrayTree<T,L>::remove_end(int x){
	int needed_size=_size-x;
	resize(needed_size); 	
};

template<typename T,typename L>
template<typename D>
void HashedArrayTree<T,L>::shift(HashedArrayTree<T,L>::iterator it,int n){
	typedef std::tuple<Cordinate,Cordinate,int>	paramater;
	typedef std::vector<paramater>			paramater_vector;
	
	int width=_tree.width(); 
	Internal::shift_functions<D,paramater_vector,T,allocator_type> util(n);

	paramater_vector param_v;
	
	Cordinate src=it._cordinate;
	Cordinate dst=src;
	int copy_size;

	while( src.distance() < size()){
		dst=util.next(src); 	
		copy_size=util.next_size(src); 
	
		paramater param=std::make_tuple(src,dst,copy_size);
		if(dst.distance()<size()){
			param_v.insert(param_v.begin(),param); 
		}
		src=util.move(1,dst); 
	}
	util.adjust(param_v); 
	std::for_each(param_v.begin(),param_v.end(),
		[this](paramater p){
			typedef typename tree::device_pointer pointer; 
			Cordinate source=		std::get<0>(p);
			Cordinate destination=	std::get<1>(p);
			int s=				std::get<2>(p); 

			pointer src_it=(this->_tree).getbranch(
					source.row() )+source.offset(); 
			pointer dst_it=(this->_tree).getbranch(
					destination.row() )+destination.offset() ; 

			this->location.MemCopy(src_it,dst_it,s*sizeof(T) ); 
		}
	);

}

template<typename T,typename L>
template<typename I>
void HashedArrayTree<T,L>::copy(I it,int n,const T& item){
	location.fill_in(it,n,item);
}
template<typename T,typename L>
template<typename I>
void HashedArrayTree<T,L>::copy(I it_in,int n,I it_out){
	location.MemCopy(it_in,it_out,n); 
}

//***********************************INTERNAL FUNCTIONS********************************************
//***********************************ITERATOR GETTERS********************************************

template<typename T,typename L>
HashedArrayTree<T,L>::iterator HashedArrayTree<T,L>::begin(){
	iterator it(_tree);
	it.initalize(0,0); 
	return it; 
};

template<typename T,typename L>
HashedArrayTree<T,L>::iterator HashedArrayTree<T,L>::end(){
	iterator it(_tree);
	it.initalize( size() );
	return it; 
};

template<typename T,typename L>
HashedArrayTree<T,L>::const_iterator HashedArrayTree<T,L>::cbegin(){
	const_iterator it=begin();
	return it;
};

template<typename T,typename L>
HashedArrayTree<T,L>::iterator HashedArrayTree<T,L>::cend(){
	const_iterator it=end();
	return it; 
};


template<typename T,typename L>
HashedArrayTree<T,L>::reverse_iterator HashedArrayTree<T,L>::rbegin(){
	reverse_iterator it(_tree);
	it.initalize(0,0); 
	return it; 
};

template<typename T,typename L>
HashedArrayTree<T,L>::reverse_iterator HashedArrayTree<T,L>::rend(){	
	reverse_iterator it(_tree);
	it.initalize( size() );
	return it; 
};

template<typename T,typename L>
HashedArrayTree<T,L>::const_reverse_iterator HashedArrayTree<T,L>::crbegin(){
	const_reverse_iterator it=rbegin();
	return it;
};

template<typename T,typename L>
HashedArrayTree<T,L>::const_reverse_iterator HashedArrayTree<T,L>::crend(){
	const_reverse_iterator it=rend();
	return it; 
};

//***********************************ITERATOR GETTERS********************************************
//***********************************OPERATORS********************************************
//deal with T!=T=>false and L!=L=>false,maybe
template<typename T,typename L>
template<typename A>
HashedArrayTree<T,L>& HashedArrayTree<T,L>::operator=(const HashedArrayTree<T,A>& other){
	
//	HashedArrayTree_Internal::Assign<T,L,A> _assign;
//	_assign(this,other);
	return *this;
};
template<typename T,typename L>
HashedArrayTree<T,L>& HashedArrayTree<T,L>::operator=(const HashedArrayTree<T,L>& other){
	
	this->_size=other._size;
	this->_tree=other._tree;
	return *this;
};

template<typename T,typename L>
bool HashedArrayTree<T,L>::operator==(const HashedArrayTree<T,L>& other)const{
	if(_size==other._size){
		return (this->_tree)==(other._tree);
	}else{
		return false;
	}
};

template<typename T,typename L>
bool HashedArrayTree<T,L>::operator!=(const HashedArrayTree<T,L>& other)const{
	return !(*this==other);
};

//***********************************OPERTORS********************************************
///***********************************INTERFACE********************************************/

template<typename T,typename L>
void	HashedArrayTree<T,L>::swap(HashedArrayTree<T,L>& other){
	_tree.swap( (other._tree) );

	std::swap(this->_size,other._size); 
};

template<typename T,typename L>
HashedArrayTree<T,L>::size_type HashedArrayTree<T,L>::size()const{
	return _size; 
};

template<typename T,typename L>
HashedArrayTree<T,L>::size_type HashedArrayTree<T,L>::capacity()const{
	return std::pow(_tree.width(),2); 
};

template<typename T,typename L>
HashedArrayTree<T,L>::size_type HashedArrayTree<T,L>::max_size()const{
	return _allocator.max_size(); 
};

template<typename T,typename L>
bool	HashedArrayTree<T,L>::empty()const{
	return _size>0;
};

template<typename T,typename L>
void HashedArrayTree<T,L>::reserve(int x){
	resize(x);
};

template<typename T,typename L>
void	HashedArrayTree<T,L>::shrink_to_fit(){
	resize( size() ); 
};

/*	
emplate<typename T,typename L>
template<class ...Args>
	HashedArrayTree<T,A>::iterator HashedArrayTree<T,A>::emplace(HashedArrayTree<T,A>::const_iterator iter, Args...){

	};
*/
template<typename T,typename L>
HashedArrayTree<T,L>::iterator HashedArrayTree<T,L>::insert(HashedArrayTree<T,L>::const_iterator iter, const T& item){
	add_end(1);
	shift<DOWN>(iter,1);	
	copy(iter,1,item);
	return iter+1;
}

template<typename T,typename L>
HashedArrayTree<T,L>::iterator HashedArrayTree<T,L>::insert(HashedArrayTree<T,L>::const_iterator iter, T&& item){ 
	add_end(1);
	shift<DOWN>(iter,1); 
	copy(iter,1,item); 
	return iter+1;
};

template<typename T,typename L>
HashedArrayTree<T,L>::iterator HashedArrayTree<T,L>::insert(HashedArrayTree<T,L>::const_iterator iter, HashedArrayTree<T,L>::size_type n, T& item){
	add_end(n); 
	shift<DOWN>(iter,n); 
	copy(iter,n,item);
	return iter+1;
}
	
template<typename T,typename L>
template<class iter>
	HashedArrayTree<T,L>::iterator HashedArrayTree<T,L>::insert(HashedArrayTree<T,L>::const_iterator iter_1, iter iter_2,  iter iter_3){
 	int n=iter_2-iter_1;
	add_end(n);
	shift<DOWN>(iter_1,n);
	copy(iter_1,n,iter_2);
	iter_1.setWidth( _tree.width() ); 
	return iter_1+1;	
};
/*
specialize for location

template<typename T,typename L>
HashedArrayTree<T,A>::iterator HashedArrayTree<T,A>::insert(HashedArrayTree<T,A>::const_iterator iter, std::initializer_list<T> list){

};
*/

template<typename T,typename L>
HashedArrayTree<T,L>::iterator HashedArrayTree<T,L>::erase(HashedArrayTree<T,L>::const_iterator iter){
	shift<UP>(iter,1); 
	remove_end(1);
	return iter-1; 
};

template<typename T,typename L>
HashedArrayTree<T,L>::iterator HashedArrayTree<T,L>::erase(HashedArrayTree<T,L>::const_iterator iter_begin , HashedArrayTree<T,L>::const_iterator iter_end){
	int n=iter_end-iter_begin; 
	shift<UP>(iter_end,n);
	remove_end(n); 
	return iter_begin; 
};


template<typename T,typename L>
void	HashedArrayTree<T,L>::clear(){
	_tree.clear(); 
	_size=0; 
};
/*
template<typename T,typename L>
template<class iter> void HashedArrayTree<T,A>::assign(iter it_1, iter it_2){

};
*/
template<typename T,typename A>
void	HashedArrayTree<T,A>::assign(std::initializer_list<T> list){
	assign(list.begin(),list.end()); 
};

template<typename T,typename L>
void	HashedArrayTree<T,L>::assign(HashedArrayTree<T,L>::size_type n, const T& item){
	clear();
	resize(n);
	copy(this->begin(),n,item); 
};

template<typename T,typename A>
HashedArrayTree<T,A>::allocator_type HashedArrayTree<T,A>::get_allocator(){
	return _allocator; 
};

template<typename T,typename A>
HashedArrayTree<T,A>::reference HashedArrayTree<T,A>::operator[](size_type x){
	return at(x);
};
template<typename T,typename A>
HashedArrayTree<T,A>::reference HashedArrayTree<T,A>::at(size_type x){
	return *(begin()+x);
};
template<typename T,typename A>
HashedArrayTree<T,A>::reference HashedArrayTree<T,A>::front(){
	return at(0);
};
template<typename T,typename A>
HashedArrayTree<T,A>::reference HashedArrayTree<T,A>::back(){
	return at(size() ); 
};

//***********************************INTERFACE ******************************************** 
//***********************************LOCKING********************************************

template<typename T,typename L>
void	HashedArrayTree<T,L>::lock(){
	_mutex.lock();
};

template<typename T,typename L>
void	HashedArrayTree<T,L>::unlock(){
	_mutex.unlock(); 
};

template<typename T,typename L>
bool	HashedArrayTree<T,L>::try_lock(){
	bool result=_mutex.try_lock(); 
	return result; 
};
//***********************************LOCKING********************************************















