#include <gtest/gtest.h>
#include <graph.cpp> 
#include <task.cpp>
#include <dataframe.cpp>

#define GRAPH_THREADS 8
#define GRAPH_SIZE 10 

#include<MacroUtilities.cpp>
#include<vector>
#include<thread>
#include<stdio.h>

template<class ... Type>
class graphTest : public ::testing::Test{
	typedef dataframe<int,int>				frame_int; 
	typedef dataframe<float,float>			frame_float; 
	typedef scheduler::task_adapter<frame_int,frame_float>	Task;
	typedef typename Task::Args				Args;  
	typedef typename Task::Msg				Msg; 

	typedef typename frame_int::value_type		int_value; 
	typedef typename frame_float::value_type	float_value; 

	struct Fun1{
		void operator()(Args& arg){
			int_value end1(1,1); 
			float_value end2(1,1); 

			std::get<0>(arg)[0]=end1;
			std::get<1>(arg)[0]=end2;	
		}; 
	};
	struct Fun2{
		void operator()(Args& arg){
			int_value end1(2,2); 
			float_value end2(2,2); 

			std::get<0>(arg)[1]=end1;
			std::get<1>(arg)[1]=end2;	
		}; 
	};

	
	int_value start_int;
	float_value start_float; 	
	frame_int fint;
	frame_float ffloat; 

	Args args; 
	Fun1 fun1;
	Fun2 fun2; 

	Task task1;
	Task task2; 

	public:
	graphTest():	start_int(0,0),
				start_float(0,0),
				fint(10,start_int),
				ffloat(10,start_float),
				args(fint,ffloat),
				task1(fun1,&args),
				task2(fun2,&args){}
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
	typedef scheduler::task_graph  Graph; 
	typedef typename Graph::node	 Node; 

	Graph graph; 

	Node t1_ptr=graph.register_task(task1);
	Node t2_ptr=graph.register_task(task2); 

	graph.start(t1_ptr); 
	graph.dependency(t1_ptr,t2_ptr); 
	graph.run();

	int_value end1i(1,1); 
	float_value end1f(1,1); 
	int_value end2i(2,2); 
	float_value end2f(2,2); 

	EXPECT_EQ(fint[0],end1i);
	EXPECT_EQ(ffloat[0],end1f); 
	EXPECT_EQ(fint[1],end2i);
	EXPECT_EQ(ffloat[1],end2f); 
}

//python:key:tests=EmptyTest MainTest
//python:template=TEST_F($graphTest<int,double,long,float>$,|tests|){this->|tests|();}

//python:start
//python:include=graph.test
#include"graph.test"
//python:end

#undef GRAPH_THREADS

