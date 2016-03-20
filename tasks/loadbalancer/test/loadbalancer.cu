#include <location.cu>
#include <gtest/gtest.h>

#define LOADBALANCER_THREADS 8
#define LOADBALANCER_SIZE 10 

#include<MacroUtilities.cpp>
#include<loadbalancer.cpp>
#include<vector>
#include<thread>
#include<stdio.h>

template<class ... Type>
class loadbalancerTest : public ::testing::Test{
	public:

//	Container global_container; 	
	DEFINE(EmptyTest,		LOADBALANCER_THREADS)
	DEFINE(MainTest,		LOADBALANCER_THREADS)
};

template<class ... Type>
void loadbalancerTest<Type...>::EmptyTest()
{

}

template<class ... Type>
void loadbalancerTest<Type...>::MainTest()
{

}



//python:key:tests=EmptyTest MainTest
//python:template=TEST_F($loadbalancerTest<int,double,long,float>$,|tests|){this->|tests|();}

//python:start
//python:include=loadbalancer.test
#include "loadbalancer.test"
//python:end

#undef LOADBALANCER_THREADS
