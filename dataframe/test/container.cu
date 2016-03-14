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
	typedef typename Container::value_type value_type;
	
//	Container global_container; 	
	
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

	Container local(10);
	Container local_empty_copy(local);

	value_type value(0,0,0,0);
	Container local2(10,value); 
		
};
template<class ... Type>
void dataframeTest<Type...>::AssignmentTest()
{
	Container local(10);
	value_type value(0,0,0,0);

	iterator it=local.begin(); 

	value_type value2=local[0]; 	
	value=*it;
	value=local[0];
	*it=value; 
};
template<class ... Type>
void dataframeTest<Type...>::EqualityTest()
{
//	EXPECT_TRUE(vector==vector);
};
template<class ... Type>
void dataframeTest<Type...>::BeginEndTest()
{

	Container local(10);	
	iterator b=local.begin();
	iterator e=local.end();
	EXPECT_TRUE(b<e);
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

	typedef typename Container::size_type size_type;	
	Container local;
	
	size_type a=local.size();
	EXPECT_EQ(a,0);
	size_type b=local.max_size();
	size_type c=local.capacity();
	EXPECT_LT(c,b); 
	EXPECT_LE(a,c); 

	bool d=local.empty();
	EXPECT_TRUE(d); 

	size_type size=10;
	for(int i=1; i<10;i++){
		size+=10*i; 	
	
		local.resize(size);	
		a=local.size();
		EXPECT_EQ(a,size);
		b=local.max_size();
		c=local.capacity();
		EXPECT_LT(c,b); 
		EXPECT_LE(a,c); 

		d=local.empty();
		EXPECT_FALSE(d); 
	}
}


//python:key:tests=EmptyTest BeginEndTest InsertTest AccessTest ModifyTest QuerryTest LockTest EqualityTest ConstructorTest AssignmentTest
//python:template=TEST_F($dataframeTest<int,float,double,long>$,|tests|){this->|tests|();}

//python:start
//python:include=container.test
#include"container.test"
//python:end

#undef DATAFRAME_THREADS




