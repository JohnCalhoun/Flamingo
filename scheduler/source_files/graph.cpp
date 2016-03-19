//graph.cpp
#ifndef GRAPH_SCHEDULER_CPP
#define GRAPH_SCHEDULER_CPP
#include <tbb/flow_graph.h> 
#include "task.cpp"

namespace scheduler {

class task_graph {
	private: 
	typedef tbb::flow::continue_msg		Msg; 
	typedef tbb::flow::broadcast_node<Msg>	Start;
	typedef tbb::flow::graph				Graph;
	
	public:
	typedef tbb::flow::continue_node<Msg>	Node; 

	task_graph():graph(),source(graph){}; 
	~task_graph(){}; 

	Graph graph; 
	Start source; 
	
	template<class ... DataFrames>
	Node* register_task(task_body<DataFrames...>);
	
	void start(Node*);
	void dependency(Node*,Node*); 	

	void run();
	void run(int);  
};

#include "graph.inl"
}//scheduler
#endif
