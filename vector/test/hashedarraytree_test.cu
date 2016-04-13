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
	DEFINE(LockTest,		HASHEDARRAYTREE_THREADS)
	DEFINE(QuerryTest,		HASHEDARRAYTREE_THREADS)
	DEFINE(InsertTest,		HASHEDARRAYTREE_THREADS)
	DEFINE(AccessTest,		HASHEDARRAYTREE_THREADS)
	DEFINE(ModifyTest,		HASHEDARRAYTREE_THREADS)
};

template<typename T,Region M>
void HashedArrayTreeTest<T,M>::InsertTest(){
	Container local;
	int s=local.size();
	local.insert(local.begin(),1);
	EXPECT_TRUE(s<local.size());
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
//	Container local_vector; 
};
template<typename T,Region M>
void HashedArrayTreeTest<T,M>::AssignmentTest(){
/*
	Container local_vector; 
	local_vector=vector;	
	EXPECT_TRUE(local_vector==vector); 
*/
};
template<typename T,Region M>
void HashedArrayTreeTest<T,M>::EqualityTest(){
//	EXPECT_TRUE(vector==vector);
};
template<typename T,Region M>
void HashedArrayTreeTest<T,M>::BeginEndTest(){
/*
	typedef typename Container::iterator iterator; 
	
	iterator b=vector.begin();
	iterator e=vector.end();
	iterator cb=vector.cbegin();
	iterator ce=vector.cbegin();
*/
};
template<typename T,Region M>
void HashedArrayTreeTest<T,M>::LockTest(){
/*
	vector.lock();
	global_host[0]++;
	vector.unlock();

	bool p=vector.try_lock();
	if(p)
		vector.unlock();
*/
}
template<typename T,Region M>
void HashedArrayTreeTest<T,M>::QuerryTest(){
/*
	typedef typename Container::size_type size;	

	size a=vector.size();
	size b=vector.max_size();
	size c=vector.capacity();
	bool d=vector.empty();
*/
}
///device**********************
const Region host=Region::host;
const Region device=Region::device;
const Region pinned=Region::pinned;
const Region unified=Region::unified;

//python:key:testsH=InsertTest AccessTest ModifyTest QuerryTest LockTest EqualityTest ConstructorTest AssignmentTest
//python:key:locationH=host device
//python:template=TEST_F($HashedArrayTreeTest<int,|locationH|>$,|testsH||locationH|){this->|testsH|();}

//python:start
//python:include=hashedarraytree.test
#include"hashedarraytree.test"
//python:end

#undef HASHEDARRAYTREE_THREADS




