//traits.cpp
#include <tuple>

namespace scheduler{

template<class ... DataFrames>
class data {
	typedef std::tuple<DataFrames...> type; 
};

template<class ... DataFrames>
class traits {
	typedef typename scheduler::data<DataFrames...>	Args;
	typedef typename std::function<void(Args)>		Function; 
	typedef typename tbb::flow::continue_msg		Msg; 
}

}//scheduler
