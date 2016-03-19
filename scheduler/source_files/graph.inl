//graph.inl

template<class ... DataFrames>
task_graph::Node* task_graph::register_task(task_body<DataFrames...> task){
	Node* new_task=new Node(graph,task);
	return new_task;
}

void task_graph::start(task_graph::Node& A){
	tbb::flow::make_edge(source,A); 
}

void task_graph::dependency(task_graph::Node& A,task_graph::Node& B){
	tbb::flow::make_edge(A,B); 
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
