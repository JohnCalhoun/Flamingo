//graph.cpp
#include <tbb/flow_graph.h> 
#include "task.cpp"

namespace scheduler {

class task_graph {
	typedef tbb::flow::graph				Graph;
	typedef tbb::flow::continue_msg		msg; 
	typedef tbb::flow::continue_node<msg>	node; 
	typedef tbb::flow:broadcast_ndoe<msg>	Start;

	task_graph(); 
	~task_graph(); 

	Graph graph; 
	Start start; 
	
	template<class ... DataFrames>
	node& register_task(task_body<DataFrames...>);
	
	void start(node&);
	void dependency(node&,node&); 	
	void remove(); 

	void run();
	void run(int);  
}

#include "graph.inl"
}//scheduler
