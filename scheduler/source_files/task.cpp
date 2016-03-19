//task.cpp
#include "traits.cpp"
#include <tuple>
#include <functional>
#include <tbb/flow_graph>

namespace scheduler {

template<class ... DataFrames> 
struct task_body {	
	typedef typename traits<DataFrames...>	traits; 
	typedef typename traits::Args			Args;
	typedef typename traits::Function		Function; 
	typedef typename traits::Msg			Msg; 
	
	task(Function funct,Args& arg): args(arg),function(funct); 
	~task(); 
	
	Args& args; 
	Function function; 

	void operator()(Msg); 
}

#include "task.inl"
}//scheduler
