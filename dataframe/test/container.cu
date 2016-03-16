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
	typedef typename Container::value_type		value_type;
	
//	Container global_container; 	
	DEFINE(AddressTest,		DATAFRAME_THREADS)
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
void dataframeTest<Type...>::AddressTest()
{
	Container local(100); 
	local.id(0);
	Container* ptr=static_cast<Container*>(dataframeBase::find(0));

	EXPECT_EQ(local,*ptr); 
}

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


}
template<class ... Type>
void dataframeTest<Type...>::ModifyTest()
{
	value_type value(1,2,3,4);
	value_type start(0,0,0,0);
	Container local(10,start); 

	local[5]=value;
	EXPECT_EQ(local[5],value); 

	iterator it=local.begin();
	*it=value; 
	EXPECT_EQ(*it,value); 	
	
	iterator it2=it+1;	
	local.insert(it2,value); 	
	EXPECT_EQ(*(local.begin()+1),value); 	

	Container local2(10,value); 
	local.insert(local.begin(),local2.begin(),local2.end()); 
	for(int i=0;i<10;i++){
		EXPECT_EQ(local[i],value); 
	}

}
template<class ... Type>
void dataframeTest<Type...>::ConstructorTest()
{
	const int size=10;

	Container local(size);
	EXPECT_EQ(local.size(),size); 

	Container local_empty_copy(local);
	EXPECT_EQ(local,local_empty_copy); 

	value_type value(0,0,0,0);
	Container local2(size,value); 
	for(int i=0;i<size;i++){
		EXPECT_EQ(local2[i],value); 
	}
};
template<class ... Type>
void dataframeTest<Type...>::AssignmentTest()
{
	const int size=10;
	Container local(size);
	for(int i=0;i<size;i++){
		value_type value(i,i,i,i);	
		local[i]=value;
		EXPECT_EQ(local[i],value);
	}
};
template<class ... Type>
void dataframeTest<Type...>::EqualityTest()
{
	Container local1(10);
	Container local2(10);
	local1=local2; 
	EXPECT_EQ(local1,local2);
	local2.resize(1); 
	EXPECT_NE(local1,local2);
	local1=local2;
	EXPECT_EQ(local1,local2);
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


//python:key:tests=EmptyTest BeginEndTest InsertTest AccessTest ModifyTest QuerryTest LockTest EqualityTest ConstructorTest AssignmentTest AddressTest
//python:template=TEST_F($dataframeTest<int,float,double,long>$,|tests|){this->|tests|();}

//python:start
//python:include=container.test
#include"container.test"
//python:end

#undef DATAFRAME_THREADS




