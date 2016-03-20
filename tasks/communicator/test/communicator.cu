#include <location.cu>
#include <gtest/gtest.h>

#define COMMUNICATOR_THREADS 8
#define COMMUNICATOR_SIZE 10 

#include<MacroUtilities.cpp>
#include<communicator.cpp>
#include<vector>
#include<thread>
#include<stdio.h>

template<class ... Type>
class communicatorTest : public ::testing::Test{
	public:

//	Container global_container; 	
	DEFINE(EmptyTest,		COMMUNICATOR_THREADS)
	DEFINE(MainTest,		COMMUNICATOR_THREADS)
};

template<class ... Type>
void communicatorTest<Type...>::EmptyTest()
{

}

template<class ... Type>
void communicatorTest<Type...>::MainTest()
{

}



//python:key:tests=EmptyTest MainTest
//python:template=TEST_F($communicatorTest<int,double,long,float>$,|tests|){this->|tests|();}

//python:start
//python:include=communicator.test
#include "communicator.test"
//python:end

#undef COMMUNICATOR_THREADS
