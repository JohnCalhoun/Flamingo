#include <location.cu>
#include <gtest/gtest.h>

#define AGENT_THREADS 8
#define AGENT_SIZE 10 

#include<MacroUtilities.cpp>
#include<agent.cpp>
#include<vector>
#include<thread>
#include<stdio.h>

template<class ... Type>
class agentTest : public ::testing::Test{
	public:

//	Container global_container; 	
	DEFINE(EmptyTest,		AGENT_THREADS)
	DEFINE(MainTest,		AGENT_THREADS)
};

template<class ... Type>
void agentTest<Type...>::EmptyTest()
{

}

template<class ... Type>
void agentTest<Type...>::MainTest()
{

}



//python:key:tests=EmptyTest MainTest
//python:template=TEST_F($agentTest<int,double,long,float>$,|tests|){this->|tests|();}

//python:start
//python:include=agent.test
#include "agent.test"
//python:end

#undef AGENT_THREADS
