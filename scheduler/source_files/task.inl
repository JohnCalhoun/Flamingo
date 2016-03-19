//task.inl

template<class ... DataFrames>
void task_body<DataFrames...>::operator()(
		task_body<dataFrames...>::msg message)
{
	function(args); 
}
