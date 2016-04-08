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
template<typename T,Memory M>
HashedArrayTree<T,M>::HashedArrayTree(){
	_size=0;
	resize(_size); 	
};

template<typename T,Memory M>
HashedArrayTree<T,M>::HashedArrayTree(const HashedArrayTree<T,M>& other){
	_size=other._size; 
	_tree=other._tree; 
};
template<typename T,Memory M>
HashedArrayTree<T,M>::HashedArrayTree(int x,T item){
	resize(x);
	insert(this->begin(),x,item);
};

template<typename T,Memory M>
HashedArrayTree<T,M>::~HashedArrayTree(){};

//***********************************CONSTRUCTORS/DESTRUCTORS********************************************
//***********************************INTERNAL FUNCTIONS********************************************
template<typename T,Memory M>
int HashedArrayTree<T,M>::calculate_width(int x){
	return std::pow(2,std::ceil( std::log2( std::sqrt(x) ) ) );
}

template<typename T,Memory M>
void HashedArrayTree<T,M>::resize(int x){
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
			cor.setTree(_tree);
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

template<typename T,Memory M>
HashedArrayTree<T,M>::iterator HashedArrayTree<T,M>::add_end(
			int x,
			HashedArrayTree<T,M>::iterator it	){
	int needed_size=_size+x;
	resize(needed_size); 

	iterator out(_tree,it);
	return out;
}

template<typename T,Memory M>
void HashedArrayTree<T,M>::remove_end(int x){
	int needed_size=_size-x;
	resize(needed_size); 	
};

template<typename T,Memory M>
template<typename D>
void HashedArrayTree<T,M>::shift(HashedArrayTree<T,M>::iterator it,int n){
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
			typedef typename tree::pointer pointer; 
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

template<typename T,Memory M>
template<typename I>
void HashedArrayTree<T,M>::copy(I it,int n,const T& item){
	location.fill_in(it,n,item);
}
template<typename T,Memory M>
template<typename I>
void HashedArrayTree<T,M>::copy(I it_in,int n,I it_out){
	location.MemCopy(it_in,it_out,n); 
}

//***********************************INTERNAL FUNCTIONS********************************************
//***********************************ITERATOR GETTERS********************************************

template<typename T,Memory M>
HashedArrayTree<T,M>::iterator HashedArrayTree<T,M>::begin(){
	iterator it(_tree);
	it.initalize(0,0); 
	return it; 
};

template<typename T,Memory M>
HashedArrayTree<T,M>::iterator HashedArrayTree<T,M>::end(){
	iterator it(_tree);
	it.initalize( size() );
	return it; 
};

template<typename T,Memory M>
HashedArrayTree<T,M>::const_iterator HashedArrayTree<T,M>::cbegin(){
	const_iterator it=begin();
	return it;
};

template<typename T,Memory M>
HashedArrayTree<T,M>::iterator HashedArrayTree<T,M>::cend(){
	const_iterator it=end();
	return it; 
};


template<typename T,Memory M>
HashedArrayTree<T,M>::reverse_iterator HashedArrayTree<T,M>::rbegin(){
	reverse_iterator it(_tree);
	it.initalize(0,0); 
	return it; 
};

template<typename T,Memory M>
HashedArrayTree<T,M>::reverse_iterator HashedArrayTree<T,M>::rend(){	
	reverse_iterator it(_tree);
	it.initalize( size() );
	return it; 
};

template<typename T,Memory M>
HashedArrayTree<T,M>::const_reverse_iterator HashedArrayTree<T,M>::crbegin(){
	const_reverse_iterator it=rbegin();
	return it;
};

template<typename T,Memory M>
HashedArrayTree<T,M>::const_reverse_iterator HashedArrayTree<T,M>::crend(){
	const_reverse_iterator it=rend();
	return it; 
};

//***********************************ITERATOR GETTERS********************************************
//***********************************OPERATORS********************************************
//deal with T!=T=>false and L!=L=>false,maybe
template<typename T,Memory M>
template<Memory A>
HashedArrayTree<T,M>& HashedArrayTree<T,M>::operator=(const HashedArrayTree<T,A>& other){
	_size=other._size;
	_tree=other._tree;
	return *this;
};

template<typename T,Memory M>
bool HashedArrayTree<T,M>::operator==(const HashedArrayTree<T,M>& other)const{
	if(_size==other._size){
		return (this->_tree)==(other._tree);
	}else{
		return false;
	}
};

template<typename T,Memory M>
bool HashedArrayTree<T,M>::operator!=(const HashedArrayTree<T,M>& other)const{
	return !(*this==other);
};

//***********************************OPERTORS********************************************
///***********************************INTERFACE********************************************/

template<typename T,Memory M>
void	HashedArrayTree<T,M>::swap(HashedArrayTree<T,M>& other){
	_tree.swap( (other._tree) );

	std::swap(this->_size,other._size); 
};

template<typename T,Memory M>
HashedArrayTree<T,M>::size_type HashedArrayTree<T,M>::size()const{
	return _size; 
};

template<typename T,Memory M>
HashedArrayTree<T,M>::size_type HashedArrayTree<T,M>::capacity()const{
	return std::pow(_tree.width(),2); 
};

template<typename T,Memory M>
HashedArrayTree<T,M>::size_type HashedArrayTree<T,M>::max_size()const{
	return _allocator.max_size(); 
};

template<typename T,Memory M>
bool	HashedArrayTree<T,M>::empty()const{
	return _size>0;
};

template<typename T,Memory M>
void HashedArrayTree<T,M>::reserve(int x){
	resize(x);
};

template<typename T,Memory M>
void	HashedArrayTree<T,M>::shrink_to_fit(){
	resize( size() ); 
};

/*	
emplate<typename T,Memory M>
template<class ...Args>
	HashedArrayTree<T,M>::iterator HashedArrayTree<T,M>::emplace(HashedArrayTree<T,M>::const_iterator iter, Args...){

	};
*/
template<typename T,Memory M>
HashedArrayTree<T,M>::iterator HashedArrayTree<T,M>::insert(HashedArrayTree<T,M>::const_iterator iter, const T& item){
	iterator new_iter=add_end(1,iter);

	shift<DOWN>(new_iter,1);	
	copy(new_iter,1,item);
	return new_iter+1;
}

template<typename T,Memory M>
HashedArrayTree<T,M>::iterator HashedArrayTree<T,M>::insert(HashedArrayTree<T,M>::const_iterator iter, T&& item){ 
	iterator new_iter=add_end(1,iter);

	shift<DOWN>(new_iter,1); 
	copy(new_iter,1,item); 
	return new_iter+1;
};

template<typename T,Memory M>
HashedArrayTree<T,M>::iterator HashedArrayTree<T,M>::insert(HashedArrayTree<T,M>::const_iterator iter, HashedArrayTree<T,M>::size_type n, T& item){
	size_type index=iter.distance; 

	iterator new_iter=add_end(n,iter); 

	shift<DOWN>(new_iter,n); 
	copy(new_iter,n,item);
	return new_iter+1;
}
	
template<typename T,Memory M>
template<class iter>
	HashedArrayTree<T,M>::iterator HashedArrayTree<T,M>::insert(HashedArrayTree<T,M>::const_iterator iter_1, iter iter_2,  iter iter_3){
 	int n=iter_2-iter_1;
	iterator new_iter_1=add_end(n,iter_1);

	shift<DOWN>(new_iter_1,n);

	copy(new_iter_1,n,iter_3);
	new_iter_1.setWidth( _tree.width() ); 
	return new_iter_1+1;	
};
/*
specialize for location

template<typename T,Memory M>
HashedArrayTree<T,M>::iterator HashedArrayTree<T,M>::insert(HashedArrayTree<T,M>::const_iterator iter, std::initializer_list<T> list){

};
*/

template<typename T,Memory M>
HashedArrayTree<T,M>::iterator HashedArrayTree<T,M>::erase(HashedArrayTree<T,M>::const_iterator iter){
	shift<UP>(iter,1); 
	remove_end(1);
	return iter-1; 
};

template<typename T,Memory M>
HashedArrayTree<T,M>::iterator HashedArrayTree<T,M>::erase(HashedArrayTree<T,M>::const_iterator iter_begin , HashedArrayTree<T,M>::const_iterator iter_end){
	int n=iter_end-iter_begin; 
	shift<UP>(iter_end,n);
	remove_end(n); 
	return iter_begin; 
};


template<typename T,Memory M>
void	HashedArrayTree<T,M>::clear(){
	_tree.clear(); 
	_size=0; 
};
/*
template<typename T,Memory M>
template<class iter> void HashedArrayTree<T,M>::assign(iter it_1, iter it_2){

};
*/
template<typename T,Memory M>
void	HashedArrayTree<T,M>::assign(std::initializer_list<T> list){
	assign(list.begin(),list.end()); 
};

template<typename T,Memory M>
void	HashedArrayTree<T,M>::assign(HashedArrayTree<T,M>::size_type n, const T& item){
	clear();
	resize(n);
	copy(this->begin(),n,item); 
};

template<typename T,Memory M>
HashedArrayTree<T,M>::allocator_type HashedArrayTree<T,M>::get_allocator(){
	return _allocator; 
};

template<typename T,Memory M>
HashedArrayTree<T,M>::reference HashedArrayTree<T,M>::operator[](size_type x){
	return at(x);
};
template<typename T,Memory M>
HashedArrayTree<T,M>::reference HashedArrayTree<T,M>::at(size_type x){
	return *(begin()+x);
};
template<typename T,Memory M>
HashedArrayTree<T,M>::reference HashedArrayTree<T,M>::front(){
	return at(0);
};
template<typename T,Memory M>
HashedArrayTree<T,M>::reference HashedArrayTree<T,M>::back(){
	return at(size() ); 
};

//***********************************INTERFACE ******************************************** 
