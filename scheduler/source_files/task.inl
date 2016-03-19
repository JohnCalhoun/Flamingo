//task.inl

template<class ... DataFrames>
void task_body<DataFrames...>::operator()(
		typename task_body<DataFrames...>::Msg message)
{
	function(*args); 
}
