#include <location.cu>
#include <gtest/gtest.h>

#define HASHEDARRAYTREE_THREADS 8
#define HASHEDARRAYTREE_SIZE 10 

#include<MacroUtilities.cpp>
#include<HashedArrayTree.cu>
#include<vector>
#include<thread>

#include<stdio.h>
//host location test
template<typename T,typename L>
class HashedArrayTreeTestHost : public ::testing::Test{
	public:
	typedef HashedArrayTree<T,L >		Container;
	typedef HashedArrayTree<T,host >	Container_host;
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
template<typename T,typename L>
class HashedArrayTreeTestDevice : public ::testing::Test{
	public:
	typedef HashedArrayTree<T,L >		Container;
	typedef HashedArrayTree<T,host >	Container_host;
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
	DEFINE(EmptyTest,	HASHEDARRAYTREE_THREADS)
};
/*template
template<typename T,typename L>
void HashedArrayTreeTestDevice<T,L>::(){

}
*/

template<typename T,typename L>
void HashedArrayTreeTestDevice<T,L>::EmptyTest(){

}


/*template
template<typename T,typename L>
void HashedArrayTreeTestHost<T,L>::(){

}
*/
template<typename T,typename L>
void HashedArrayTreeTestHost<T,L>::InsertTest(){
	Container local;
	int s=local.size();
	local.insert(local.begin(),1);
	EXPECT_TRUE(s<local.size());
}
template<typename T,typename L>
void HashedArrayTreeTestHost<T,L>::AccessTest(){
	auto it=global_host.begin();	
	int r=*it;
	EXPECT_EQ(r,0); 
}
template<typename T,typename L>
void HashedArrayTreeTestHost<T,L>::ModifyTest(){
/*	Container local;
	local=global_host;

	for(int i=0; i<HASHEDARRAYTREE_SIZE; i++){
		EXPECT_EQ(local[i],i);
		local[i]=i+1;
	}
	for(int i=0; i<HASHEDARRAYTREE_SIZE; i++){
		EXPECT_EQ(local[i],i+1);
	}
*/
}
template<typename T,typename L>
void HashedArrayTreeTestHost<T,L>::ConstructorTest(){
	Container local_vector; 
};
template<typename T,typename L>
void HashedArrayTreeTestHost<T,L>::AssignmentTest(){

	Container local_vector; 
	local_vector=vector;	
	EXPECT_TRUE(local_vector==vector); 

};
template<typename T,typename L>
void HashedArrayTreeTestHost<T,L>::EqualityTest(){
	EXPECT_TRUE(vector==vector);
};
template<typename T,typename L>
void HashedArrayTreeTestHost<T,L>::BeginEndTest(){

	typedef typename Container::iterator iterator; 
	
	iterator b=vector.begin();
	iterator e=vector.end();
	iterator cb=vector.cbegin();
	iterator ce=vector.cbegin();

};
template<typename T,typename L>
void HashedArrayTreeTestHost<T,L>::LockTest(){

	vector.lock();
	global_host[0]++;
	vector.unlock();

	bool p=vector.try_lock();
	if(p)
		vector.unlock();

}
template<typename T,typename L>
void HashedArrayTreeTestHost<T,L>::QuerryTest(){

	typedef typename Container::size_type size;	

	size a=vector.size();
	size b=vector.max_size();
	size c=vector.capacity();
	bool d=vector.empty();

}
///device**********************


//python:key:testsH=InsertTest AccessTest ModifyTest QuerryTest LockTest EqualityTest ConstructorTest AssignmentTest
//python:key:locationH=host device pinned
//python:template=TEST_F($HashedArrayTreeTestHost<int,|locationH|>$,|testsH||locationH|){this->|testsH|();}

//pthon:key:testsD=EmptyTest
//pthon:key:locationD=unified pinned device
//pthon:template=TEST_F($HashedArrayTreeTestDevice<int,|locationD|>$,|testsD||locationD|){this->|testsD|();}




//python:start
//python:include=hashedarraytree.test
#include"hashedarraytree.test"
//python:end

#undef HASHEDARRAYTREE_THREADS




