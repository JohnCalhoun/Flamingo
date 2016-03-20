//task.cpp
#ifndef TASK_SCHEDULER_CPP
#define TASK_SCHEDULER_CPP
#include "traits.cpp"
#include <tuple>
#include <functional>
#include <tbb/flow_graph.h>

namespace scheduler {

template<class ... DataFrames> 
struct taskBase {	
	typedef traits<DataFrames...>			Traits; 
	typedef typename Traits::Args			Args;
	typedef typename Traits::Function		Function; 
	typedef typename Traits::Msg			Msg; 
	
	virtual void operator()(Msg)=0; 
};

template<class ... DataFrames> 
struct task_adapter : public taskBase<DataFrames...> {	
	typedef typename taskBase<DataFrames...>::Traits		Traits; 
	typedef typename taskBase<DataFrames...>::Args		Args; 
	typedef typename taskBase<DataFrames...>::Function	Function; 
	typedef typename taskBase<DataFrames...>::Msg		Msg; 
	
	task_adapter(Function funct,Args* arg): args(arg),function(funct){}; 
	~task_adapter(){}; 
	
	Args* args; 
	Function function; 

	void operator()(Msg); 
};

#include "task.inl"
}//scheduler
#endif
