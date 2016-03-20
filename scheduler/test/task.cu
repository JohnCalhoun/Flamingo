#include <gtest/gtest.h>

#define TASK_THREADS 8
#define TASK_SIZE 10 

#include<MacroUtilities.cpp>
#include<vector>
#include<thread>
#include<stdio.h>
#include "task.cpp"
#include "dataframe.cpp"

template<class ... Type>
class taskTest : public ::testing::Test{
	public:
	
//	Container global_container; 	
	DEFINE(EmptyTest,	TASK_THREADS)
	DEFINE(MainTest,	TASK_THREADS)
};

template<class ... Type>
void taskTest<Type...>::EmptyTest()
{
}

template<class ... Type>
void taskTest<Type...>::MainTest()
{
	typedef dataframe<int,int>				frame_int; 
	typedef dataframe<float,float>			frame_float; 
	typedef scheduler::task_adapter<frame_int,frame_float>	Task;
	typedef typename Task::Args				Args;  
	typedef typename Task::Msg				Msg; 

	typedef typename frame_int::value_type		int_value; 
	typedef typename frame_float::value_type	float_value; 
	
	int_value start_int(0,0); 
	float_value start_float(0,0); 	
	int_value end_int(1,1); 
	float_value end_float(1,1); 
	
	struct Fun{
		void operator()(Args& arg){
			int_value end1(1,1); 
			float_value end2(1,1); 

			std::get<0>(arg)[0]=end1;
			std::get<1>(arg)[0]=end2;	
		}; 
	};
	Fun fun;
	
	frame_int fint(10,start_int);
	frame_float ffloat(10,start_float); 

	Args arg(fint,ffloat);
	Task task(	fun,
				&arg);
	Msg msg; 
	task(msg);

	EXPECT_EQ(fint[0],end_int);
	EXPECT_EQ(ffloat[0],end_float); 
}

//python:key:tests=EmptyTest MainTest
//python:template=TEST_F($taskTest<int,double,long,float>$,|tests|){this->|tests|();}

//python:start
//python:include=task.test
#include"task.test"
//python:end

#undef TASK_THREADS




