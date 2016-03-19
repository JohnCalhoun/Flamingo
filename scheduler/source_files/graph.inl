//graph.inl

task_graph::task_graph(){};
task_graph::~task_graph(){};

template<class ... DataFrames>
task_graph::node& task_graph::register_task(task_body<DataFrames...> task){
	node new_task(graph,task);
	return node;
}

template<class ... DataFrames>
void task_graph::start(task_graph<DataFrames...>::node& A){
	tbb::make_edge(start,A); 
}

void task_graph::dependency(task_graph::node& A,task_graph::node& B){
	tbb::make_edge(A,B); 
}
void task_graph::run(){
	start.try_put(msg()); 
	graph.wait_for_all(); 
}
void task_graph::run(int x){
	for(int i=0;i<x;i++){
		start.try_put(msg());
		graph.wait_for_all(); 
	} 
}
