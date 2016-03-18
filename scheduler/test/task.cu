#include <gtest/gtest.h>

#define TASK_THREADS 8
#define TASK_SIZE 10 

#include<MacroUtilities.cpp>
#include<vector>
#include<thread>
#include<stdio.h>

template<class ... Type>
class taskTest : public ::testing::Test{
	public:
	
//	Container global_container; 	
	DEFINE(EmptyTest,		TASK_THREADS)
};

template<class ... Type>
void taskTest<Type...>::EmptyTest()
{
}

//python:key:tests=EmptyTest 
//python:template=TEST_F($taskTest<int,double,long,float>$,|tests|){this->|tests|();}

//python:start
//python:include=task.test
#include"task.test"
//python:end

#undef TASK_THREADS




