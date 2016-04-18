//internal.H
#ifndef INTERNAL_H
#define INTERNAL_H

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
	template<typename T,typename L>
	class Cordinate{	
		public:
		typedef Tree<T,L>				tree;
		typedef typename tree::pointer	pointer;
		typedef typename tree::iterator	iterator;	
		typedef typename tree::size_type	size_type; 	

		Cordinate(const tree& t):
				Cordinate(0,0)
				{setTree(t);};			  
		Cordinate(const tree& t,size_type d):
				Cordinate(t) 
				{setDistance(d);};
		Cordinate(const tree& t,size_type x,size_type y):
				Cordinate(x,y)
				{setTree(t);};


		__both__ Cordinate(size_type x, size_type y):
				Cordinate()
				{set(x,y);}; 	
		__both__ Cordinate(size_type d):
				Cordinate()
				{setDistance(d);};
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
			    void setTree(const tree& t);
		__both__ void set(size_type x, size_type y);
		__both__ bool operator<(Cordinate other);
		__both__ bool operator>(Cordinate other);
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

	struct UP{
		typedef std::minus<int> operation; 	
		
		template<typename U>
			void operator()(U& vector); 
	};
	struct DOWN{
		typedef std::plus<int> operation;
		
		template<typename U>
			void operator()(U& vector);
	}; 

	template<typename D,typename V,typename T,typename L>
	class shift_functions{
		public:
		typedef typename D::operation operation; 
		typedef Cordinate<T,L>	cordinate;

		operation		op; 
		D			direction; 
		int			increment;

		shift_functions():increment(0){};
		shift_functions(int n):increment(n){};

		void set(int);
		int next_size(cordinate);
		cordinate next(cordinate);
		cordinate move(int,cordinate);
		void adjust(V&);
	};

	//*********************Directions**********************
	struct forward{
		struct Op{
			__both__ int operator()(int x, int y)const{
				return x+y;
			}
		}op;

		struct Comp{
			__both__ bool operator()(int* x,int* y)const{
				return x<y;
			}
		}comp;
	};
	struct reverse{
		struct Op{
			__both__ int operator()(int x, int y)const{
				return x-y;
			}
		}op;

		struct Comp{
			__both__ int operator()(int* x, int* y)const{
				return x<y;
			}
		}comp;
	};
	//********************Directions***********************
	template<typename pointer,unsigned int blockSize>
	__global__ void tree_leave_equality(	pointer p_1,
									pointer p_2,
									bool* result, 
									int size);

	/************************************equality operator******************/
	template<typename A,typename B,typename C,typename D>
	struct Equality_false {
		bool operator()(const Tree<A,B>& tree_1, const Tree<C,D>& tree_2);
	};
	template<typename T,typename L>
	struct Equality_device {
		bool operator()(const Tree<T,L>& tree_1, const Tree<T,L>& tree_2);
	};
	template<typename T,typename L>
	struct Equality_host {
		bool operator()(const Tree<T,L>& tree_1, const Tree<T,L>& tree_2);
	};
}; //end HashedArrayTree_Internal
#include<internal.cu>

};
};


#undef __both__
#endif



