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
	DEFINE(QuerryTest,		DATAFRAME_THREADS)
	DEFINE(InsertTest,		DATAFRAME_THREADS)
	DEFINE(AccessTest,		DATAFRAME_THREADS)
	DEFINE(ModifyTest,		DATAFRAME_THREADS)
	DEFINE(EraseTest,		DATAFRAME_THREADS)
	DEFINE(SwapTest,		DATAFRAME_THREADS)
	DEFINE(ConstTest,		DATAFRAME_THREADS)
	DEFINE(EmptyTest,		DATAFRAME_THREADS)
};

template<class ... Type>
void dataframeTest<Type...>::AddressTest()
{
	Container local(10); 
	local.id(0);
	Container* ptr=static_cast<Container*>(dataframeBase::find(0));

	EXPECT_EQ(local,*ptr); 
}
template<class ... Type>
void dataframeTest<Type...>::EraseTest()
{
	int size=10; 
	value_type one(1,1,1,1);
	value_type two(2,2,2,2);
	Container local(size,one);
	Container local2(size,one);
	Container local3(size,one);

	local.erase(local.begin());
	EXPECT_EQ(local.size(),size-1); 

	local.erase(local.begin(),local.begin()+1);
	EXPECT_EQ(local2.size(),size-2); 

	local3.push_back(two);
	EXPECT_EQ(local3.back(),two); 
	local3.pop_back(); 	
	EXPECT_EQ(local3.back(),one);
}

template<class ... Type>
void dataframeTest<Type...>::ConstTest()
{
	typedef typename Container::const_iterator const_iterator;

	value_type value(5,6,7,8);
	Container local(10,value);
 
	const_iterator itbegin=local.cbegin();
	const_iterator itend=local.cend(); 

	EXPECT_LT(itbegin,itend); 
}
template<class ... Type>
void dataframeTest<Type...>::EmptyTest()
{

}
template<class ... Type>
void dataframeTest<Type...>::SwapTest()
{
	value_type one(1,1,1,1);
	value_type two(2,2,2,2);
	int size=10;

	Container ones(size,one);
	Container twos(size,two);

	ones.swap(twos); 
	for(int i=0;i<size;i++){
		EXPECT_EQ(ones[i],two); 
		EXPECT_EQ(twos[i],one);
	}
}

template<class ... Type>
void dataframeTest<Type...>::InsertTest()
{
	const int size=10; 

	value_type value(1,2,3,4);
	value_type start(0,0,0,0);
	Container local(size,start); 

	for(int off=0;off<size;off++){
		iterator it=local.begin()+off;
		local.insert(it,value); 	
		EXPECT_EQ(*(local.begin()+off),value); 	
	}

	Container local2(size,value); 
	local.insert(local.begin(),local2.begin(),local2.end()); 
	for(int i=0;i<size;i++){
		EXPECT_EQ(local[i],value); 
	}
}

template<class ... Type>
void dataframeTest<Type...>::AccessTest()
{
	value_type value(1,2,3,4);
	value_type other(5,6,7,8);
	Container local(10,value); 

	EXPECT_EQ(local.front(),value);
	EXPECT_EQ(local.back(),value); 
	EXPECT_EQ(local.at(4),value); 
	EXPECT_EQ(local[4],value); 

	local.front()=other;
	EXPECT_EQ(local.front(),other);
	local.back()=other;
	EXPECT_EQ(local.back(),other); 
	local.at(4)=other;
	EXPECT_EQ(local.at(4),other); 
	local[5]=other;
	EXPECT_EQ(local[5],other); 
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
	it=local.begin()+1; 
	*it=value;
	EXPECT_EQ(*it,value); 
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

	Container local3(local2.begin(),local2.end()); 
};
template<class ... Type>
void dataframeTest<Type...>::AssignmentTest()
{
	value_type value(1,2,3,4);
	value_type start(0,0,0,0);
	Container local(10,start); 

	const int size=10;
	Container local2(size);
	for(int i=0;i<size;i++){
		value_type value(i,i,i,i);	
		local[i]=value;
		EXPECT_EQ(local[i],value);
	}

	int count=4; 
	local2.assign(count,value);
	for(int i=0;i<count;i++){
		local2[i]=value; 
	}

	local2.assign(local.begin(),local.end());
	for(int i=0;i<10;i++){
		local2[i]=start; 
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
	EXPECT_TRUE(local1!=local2); 

	local1=local2;
	EXPECT_EQ(local1,local2);
	EXPECT_TRUE(local1==local2);
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


//python:key:tests=ConstTest SwapTest EmptyTest BeginEndTest InsertTest AccessTest ModifyTest QuerryTest EqualityTest ConstructorTest AssignmentTest AddressTest
//python:template=TEST_F($dataframeTest<int,double,long,float>$,|tests|){this->|tests|();}

//python:start
//python:include=container.test
#include"container.test"
//python:end

#undef DATAFRAME_THREADS




