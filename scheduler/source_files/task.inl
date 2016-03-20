//task.inl

template<class ... DataFrames>
void task_adapter<DataFrames...>::operator()(
		typename task_adapter<DataFrames...>::Msg message)
{
	function(*args); 
}
