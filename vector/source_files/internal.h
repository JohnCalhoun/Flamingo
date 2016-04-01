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

namespace Internal {
	//********************************Cordinate*****************
	template<typename SRC, typename DST>
	class copyToDevice; 
	
	template<typename T,typename L>
	class Cordinate{	
		public:
		typedef Tree<T,L> tree;
		typedef typename tree::device_pointer pointer;
		
		__both__ Cordinate(int x, int y):
					_data(x,y){}; 
		
		__both__ Cordinate(int x, int y, tree* t):
					_data(x,y),tree_ptr(t){}; 
		
		__both__ Cordinate(int x, int y, tree t):
					_data(x,t),tree_ptr(t){};
		__both__ Cordinate():
					_data(0,0),tree_ptr(NULL){};
		struct data{
			__both__ data(int x,int y):first(x),second(y){};
			int first;
			int second;
		}; 
		data _data;

		tree*  tree_ptr;
			
		__both__ int width()const;
		__both__ int row()const;
		__both__ int offset()const;
		__both__ int distance()const;

		__both__ void setRow(int x);
		__both__ void setOffset(int x);
		__both__ void setDistance(int x);
		__both__ void setTree(tree* t);
		__both__ void set(int x, int y);
		__both__ bool operator<(Cordinate other);
		__both__ bool operator>(Cordinate other);
		__both__ pointer access();
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


#undef __both__
#endif



