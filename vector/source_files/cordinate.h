//internal.H
#ifndef CORDINATE_H
#define CORDINATE_H

#include<cuda.h>
#include<Tree.cu> 
#include<tuple>
#include<functional>
#include<algorithm>
#include<location.cu>

#define __both__ __host__ __device__ 
namespace Flamingo {
namespace Vector {
namespace Internal {
	//********************************Cordinate*****************
	template<typename T>
	class Cordinate{	
		public:
		typedef T*					pointer;
		typedef pointer*				iterator;	
		typedef size_t					size_type; 	

		template<typename A>
		Cordinate(const Tree<T,A>& t):
				Cordinate(0,0)
				{setTree(t);}
		template<typename A>			  
		Cordinate(const Tree<T,A>& t,size_type d):
				Cordinate(t) 
				{setDistance(d);}

		template<typename A>
		Cordinate(const Tree<T,A>& t,size_type x,size_type y):
				Cordinate(x,y)
				{setTree(t);}


		__both__ Cordinate(size_type x, size_type y):
				Cordinate()
				{set(x,y);}; 	
		__both__ Cordinate(size_type d):
				Cordinate()
				{setDistance(d);};
		__both__ Cordinate(T* t):
				_data(0,0),
				begin(&t),
				end(&t+1){}; 
		__both__ Cordinate():
				_data(0,0),
				begin(NULL),
				end(NULL){};

		struct data{
			__both__ data(size_type x,size_type y):first(x),second(y){};
			size_type first;
			size_type second;
		}; 
		
		__both__ size_type width()const;
		__both__ size_type row()const;
		__both__ size_type offset()const;
		__both__ size_type distance()const;

		__both__ void setRow(size_type x);
		__both__ void setOffset(size_type x);
		__both__ void setDistance(size_type x);

			    template<typename A>
			    void setTree(const Tree<T,A>& t);

		__both__ void set(size_type x, size_type y);
		__both__ bool operator<(Cordinate<T> other);
		__both__ bool operator>(Cordinate<T> other);
		__both__ pointer access();

		__both__ Cordinate& operator++();
		__both__ Cordinate  operator++(int);
		__both__ Cordinate& operator--();
		__both__ Cordinate  operator--(int);

		__both__ Cordinate & operator+=(size_type);
		__both__ Cordinate & operator-=(size_type);

		__both__ Cordinate operator+(size_type);
		__both__ Cordinate operator-(size_type);

		__both__ size_type operator-(Cordinate);


		data _data;
		iterator begin;
		iterator end; 	
	};
}; //end HashedArrayTree_Internal
#include<cordinate.cpp>

};
};


#undef __both__
#endif



