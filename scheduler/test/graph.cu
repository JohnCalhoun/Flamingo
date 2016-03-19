#include <gtest/gtest.h>
#include <graph.cpp> 

#define GRAPH_THREADS 8
#define GRAPH_SIZE 10 

#include<MacroUtilities.cpp>
#include<vector>
#include<thread>
#include<stdio.h>

template<class ... Type>
class graphTest : public ::testing::Test{
	public:
	
//	Container global_container; 	
	DEFINE(EmptyTest,		GRAPH_THREADS)
	DEFINE(MainTest,		GRAPH_THREADS)
};

template<class ... Type>
void graphTest<Type...>::EmptyTest()
{
}

template<class ... Type>
void graphTest<Type...>::MainTest()
{
}

//python:key:tests=EmptyTest MainTest
//python:template=TEST_F($graphTest<int,double,long,float>$,|tests|){this->|tests|();}

//python:start
//python:include=graph.test
#include"graph.test"
//python:end

#undef GRAPH_THREADS




