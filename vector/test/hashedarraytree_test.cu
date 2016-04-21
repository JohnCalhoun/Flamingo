#include <location.cu>
#include <gtest/gtest.h>

#define HASHEDARRAYTREE_THREADS 8
#define HASHEDARRAYTREE_SIZE 40 

#include<MacroUtilities.cpp>
#include<HashedArrayTree.cu>
#include<vector>
#include<thread>

#include<stdio.h>
//host location test
using namespace Flamingo::Memory;
using namespace Flamingo::Vector; 

template<typename T,Region M>
class HashedArrayTreeTest : public ::testing::Test{
	public:
	typedef HashedArrayTree<T,M >		Container;
	typedef HashedArrayTree<T,Region::host >	Container_host;
	typedef typename Container_host::iterator	host_iterator;
	Container_host global_host;
	Container vector;
	virtual void SetUp(){
		int i;
		global_host.resize(HASHEDARRAYTREE_SIZE);
		for(i=0; i<HASHEDARRAYTREE_SIZE; i++){
			global_host[i]=i;
		}
	}
	DEFINE(ConstructorTest,	HASHEDARRAYTREE_THREADS)
	DEFINE(AssignmentTest,	HASHEDARRAYTREE_THREADS)	
	DEFINE(EqualityTest,	HASHEDARRAYTREE_THREADS)
	DEFINE(BeginEndTest,	HASHEDARRAYTREE_THREADS)
	DEFINE(QuerryTest,		HASHEDARRAYTREE_THREADS)
	DEFINE(InsertTest,		HASHEDARRAYTREE_THREADS)
	DEFINE(AccessTest,		HASHEDARRAYTREE_THREADS)
	DEFINE(ModifyTest,		HASHEDARRAYTREE_THREADS)
	DEFINE(CopyTest,		HASHEDARRAYTREE_THREADS)
};

template<typename T,Region M>
void HashedArrayTreeTest<T,M>::CopyTest(){
	const int start_size=40; 
	Container local(start_size,0);
	std::array<int,start_size> array; 
	array.fill(1); 

//	local.copy_to_array( array.begin() ); 

//	for(int i=0; i<start_size; i++){
//		EXPECT_EQ( array[i],local[i]);
//	}
}

template<typename T,Region M>
void HashedArrayTreeTest<T,M>::InsertTest(){
	for(	int start_size=20; start_size<5;start_size++){
		Container local(start_size);
		int s=local.size();
		local.insert(local.begin(),1);
		EXPECT_EQ(	local.size(),(start_size+1) );
	}
}
template<typename T,Region M>
void HashedArrayTreeTest<T,M>::AccessTest(){
	auto it=global_host.begin();	
	int r=*it;
	EXPECT_EQ(r,0); 
}
template<typename T,Region M>
void HashedArrayTreeTest<T,M>::ModifyTest(){
	Container local;
	local=global_host;

	for(int i=0; i<HASHEDARRAYTREE_SIZE; i++){
		EXPECT_EQ(local[i],global_host[i]);
		local[i]=i+1;
	}
	for(int j=0; j<HASHEDARRAYTREE_SIZE; j++){
		EXPECT_EQ(local[j],j+1);
	}
}
template<typename T,Region M>
void HashedArrayTreeTest<T,M>::ConstructorTest(){
	Container local_vector;

	Container vector1(10); 
	Container vector2(vector1); 

	EXPECT_TRUE(vector1==vector2); 
};
template<typename T,Region M>
void HashedArrayTreeTest<T,M>::AssignmentTest(){
	Container local_vector; 
	local_vector=vector;	
	EXPECT_TRUE(local_vector==vector); 
};
template<typename T,Region M>
void HashedArrayTreeTest<T,M>::EqualityTest(){
	EXPECT_TRUE(vector==vector);
};
template<typename T,Region M>
void HashedArrayTreeTest<T,M>::BeginEndTest(){

	typedef typename Container::iterator iterator; 
	
	iterator b=vector.begin();
	iterator e=vector.end();
	EXPECT_TRUE( b <= e); 

	iterator cb=vector.cbegin();
	iterator ce=vector.cbegin();
	EXPECT_TRUE( cb <= ce); 
};
template<typename T,Region M>
void HashedArrayTreeTest<T,M>::QuerryTest(){

	typedef typename Container::size_type size;	

	size a=vector.size();
	size b=vector.max_size();
	size c=vector.capacity();
	bool d=vector.empty();
}
///device**********************
const Region host=Region::host;
const Region device=Region::device;
const Region pinned=Region::pinned;
const Region unified=Region::unified;

//python:key:testsH=CopyTest InsertTest AccessTest ModifyTest QuerryTest EqualityTest ConstructorTest AssignmentTest
//python:key:locationH=host device pinned unified
//python:template=TEST_F($HashedArrayTreeTest<int,|locationH|>$,|testsH||locationH|){this->|testsH|();}

//python:start
//python:include=hashedarraytree.test
#include"hashedarraytree.test"
//python:end

#undef HASHEDARRAYTREE_THREADS




