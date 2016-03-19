//graph.cpp
#include <tbb/flow_graph.h> 
#include "task.cpp"

namespace scheduler {

class task_graph {
	typedef tbb::flow::graph				Graph;
	typedef tbb::flow::continue_msg		Msg; 
	typedef tbb::flow::continue_node<Msg>	Node; 
	typedef tbb::flow::broadcast_node<Msg>	Start;

	task_graph():graph(),source(graph){}; 
	~task_graph(){}; 

	Graph graph; 
	Start source; 
	
	template<class ... DataFrames>
	Node* register_task(task_body<DataFrames...>);
	
	void start(Node&);
	void dependency(Node&,Node&); 	

	void run();
	void run(int);  
};

#include "graph.inl"
}//scheduler
