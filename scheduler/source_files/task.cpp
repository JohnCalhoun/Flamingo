//task.cpp
#ifndef TASK_SCHEDULER_CPP
#define TASK_SCHEDULER_CPP
#include "traits.cpp"
#include <tuple>
#include <functional>
#include <tbb/flow_graph.h>

namespace scheduler {

template<class ... DataFrames> 
struct task_body {	
	typedef traits<DataFrames...>			traits; 
	typedef typename traits::Args			Args;
	typedef typename traits::Function		Function; 
	typedef typename traits::Msg			Msg; 
	
	task_body(Function funct,Args* arg): args(arg),function(funct){}; 
	~task_body(){}; 
	
	Args* args; 
	Function function; 

	virtual void operator()(Msg); 
};

#include "task.inl"
}//scheduler
#endif
