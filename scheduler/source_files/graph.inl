//graph.inl

template<typename T>
task_graph::node task_graph::register_task(T task){
	node_raw* new_ptr=new node_raw(graph,task);
	node new_task(new_ptr); 
	return new_task;
}

void task_graph::start(task_graph::node A){
	tbb::flow::make_edge(source,*A); 
}

void task_graph::dependency(task_graph::node A,task_graph::node B){
	tbb::flow::make_edge(*A,*B); 
}
void task_graph::run(){
	source.try_put(Msg()); 
	graph.wait_for_all(); 
}
void task_graph::run(int x){
	for(int i=0;i<x;i++){
		source.try_put(Msg());
		graph.wait_for_all(); 
	} 
}
