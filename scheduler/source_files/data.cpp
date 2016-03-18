//data.cpp
#include <dataframe.cpp>
#include <vector>
namespace scheduler{

template<class ... Data>
class data{
	typename dataframebase::Key ID; 
	typename std::tuple<Data...> Frames; 

	Frames frames; 
	
	data(Frames frame):frames(frame);
	data(const data&)
	~data(); 	
	template<int n>
	get(); 
}

}//scheduler
