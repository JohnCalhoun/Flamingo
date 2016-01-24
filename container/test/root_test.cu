#include <location.cu>
#include <gtest/gtest.h>

#include<MacroUtilities.cpp>
#include<allocator.cu>
#include<root.h>
#include<vector>
#include<thread>
#include<stdio.h>
#define ROOT_THREADS 1
#define WIDTH 10

template<typename T, typename L>
class RootTest : public ::testing::Test{
	public: 
	typedef typename allocation_policy<T,L>::allocator Allocator; 
	typedef typename Allocator::pointer pointer;
	typedef Root<T,Allocator,L>	root;

	Allocator allocator; 
	root _root1; 
	root _root2;

	virtual void SetUp(){
		_root1.resize(WIDTH);
		_root2.resize(WIDTH+1); 
	}
	DEFINE(ResizeTest,		ROOT_THREADS)
	DEFINE(ResizeClearTest,	ROOT_THREADS)
	DEFINE(GetSetTest,		ROOT_THREADS)
	DEFINE(SwapTest,		ROOT_THREADS)
	DEFINE(AssignTest,		ROOT_THREADS)
};
template<typename T,typename L>
void RootTest<T,L>::ResizeTest(){
	root rootLocal;
	rootLocal.resize(7);
};
template<typename T,typename L>
void RootTest<T,L>::GetSetTest(){
	pointer p=allocator.allocate(WIDTH*sizeof(T)); 

	_root1.set(p,0);
	ASSERT_EQ(p,_root1.get(0) );
};
template<typename T,typename L>
void RootTest<T,L>::ResizeClearTest(){
	_root1.clear();
	ASSERT_EQ(_root1.size(),0);
};
template<typename T,typename L>
void RootTest<T,L>::AssignTest(){
	root rootLocal;
	rootLocal.resize(WIDTH+1);
	
	pointer p=allocator.allocate(WIDTH*sizeof(T)); 
	_root1.set(p,0);

	rootLocal=_root1;
};
template<typename T,typename L>
void RootTest<T,L>::SwapTest(){
//	std::swap(_root1,_root2);
//	ASSERT_EQ(_root1.size(),WIDTH+1);
//	ASSERT_EQ(_root2.size(),WIDTH);
};

//python:key:tests=ResizeTest ResizeClearTest GetSetTest SwapTest AssignTest
//python:key:location=host
//python:key:type=int float
//
//python:template=TEST_F($RootTest<|type|,|location|>$,|tests||location|){this->|tests|();}
//
//python:start
//python:include=root.test
#include"root.test"
//python:end

#undef ROOT_THREADS




