//graph.cpp
#include <tbb/flow_graph.h> 
#include "task.cpp"

class task_graph {
	typedef tbb::flow::graph Graph;

	task_graph(); 
	~task_graph(); 

	Graph graph; 
	
	void add(task);
	void remove(); 

	void run();
	void run(int);  
}
