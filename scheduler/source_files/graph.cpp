//graph.cpp
#ifndef GRAPH_SCHEDULER_CPP
#define GRAPH_SCHEDULER_CPP
#include <tbb/flow_graph.h> 
#include "task.cpp"
#include <memory>
namespace scheduler {

class task_graph {
	private: 
	typedef tbb::flow::continue_msg		Msg; 
	typedef tbb::flow::broadcast_node<Msg>	Start;
	typedef tbb::flow::graph				Graph;
	typedef tbb::flow::continue_node<Msg>	node_raw; 

	public:
	typedef std::shared_ptr<node_raw>		node;

	task_graph():graph(),source(graph){}; 
	~task_graph(){}; 

	Graph graph; 
	Start source; 
	
	template<typename T>
	node register_task(T);
	
	void start(node);
	void dependency(node,node); 	

	void run();
	void run(int);  
};

#include "graph.inl"
}//scheduler
#endif
