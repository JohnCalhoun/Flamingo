#include <location.cu>
#include <gtest/gtest.h>

#define DATAFRAME_THREADS 8
#define DATAFRAME_SIZE 10 

#include<MacroUtilities.cpp>
#include<dataframe.cpp>
#include<vector>
#include<thread>
#include<stdio.h>

template<class ... Type>
class dataframeTest : public ::testing::Test{
	public:
	typedef dataframe<Type... >				Container;
	typedef typename Container::iterator		iterator;
	typedef typename Container::value_type		element; 
	

	
	DEFINE(ConstructorTest,	DATAFRAME_THREADS)
	DEFINE(AssignmentTest,	DATAFRAME_THREADS)	
	DEFINE(EqualityTest,	DATAFRAME_THREADS)
	DEFINE(BeginEndTest,	DATAFRAME_THREADS)
	DEFINE(LockTest,		DATAFRAME_THREADS)
	DEFINE(QuerryTest,		DATAFRAME_THREADS)
	DEFINE(InsertTest,		DATAFRAME_THREADS)
	DEFINE(AccessTest,		DATAFRAME_THREADS)
	DEFINE(ModifyTest,		DATAFRAME_THREADS)
	DEFINE(EmptyTest,		DATAFRAME_THREADS)
};

template<class ... Type>
void dataframeTest<Type...>::EmptyTest()
{

}

template<class ... Type>
void dataframeTest<Type...>::InsertTest()
{
/*	Container local;
	int s=local.size();
	local.insert(local.begin(),element);
	EXPECT_TRUE(s<local.size());
*/
}

template<class ... Type>
void dataframeTest<Type...>::AccessTest()
{
/*
*/
}
template<class ... Type>
void dataframeTest<Type...>::ModifyTest()
{
/*	
*/
}
template<class ... Type>
void dataframeTest<Type...>::ConstructorTest()
{
//	Container local; 
};
template<class ... Type>
void dataframeTest<Type...>::AssignmentTest()
{
/*
	Container local;
	container other;
	
	//
	
	other=local;	
	EXPECT_TRUE(other==local); 
*/
};
template<class ... Type>
void dataframeTest<Type...>::EqualityTest()
{
//	EXPECT_TRUE(vector==vector);
};
template<class ... Type>
void dataframeTest<Type...>::BeginEndTest()
{
/*
	typedef typename Container::iterator iterator; 
	
	iterator b=vector.begin();
	iterator e=vector.end();
	iterator cb=vector.cbegin();
	iterator ce=vector.cbegin();
*/
};
template<class ... Type>
void dataframeTest<Type...>::LockTest()
{
/*
	vector.lock();
	global_host[0]++;
	vector.unlock();

	bool p=vector.try_lock();
	if(p)
		vector.unlock();
*/
}
template<class ... Type>
void dataframeTest<Type...>::QuerryTest()
{
/*
	typedef typename Container::size_type size;	

	size a=vector.size();
	size b=vector.max_size();
	size c=vector.capacity();
	bool d=vector.empty();
*/
}


//python:key:tests=EmptyTest InsertTest AccessTest ModifyTest QuerryTest LockTest EqualityTest ConstructorTest AssignmentTest
//python:template=TEST_F($dataframeTest<int,float,double,long>$,|tests|){this->|tests|();}

//python:start
//python:include=container.test
#include"container.test"
//python:end

#undef DATAFRAME_THREADS




