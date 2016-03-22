//agent.cpp
#include<task.cpp>
#include <functional>

template<class DataFrame>
class agent : public taskBase<DataFrame> {
	private:
	typedef std::ref_wrapper<DataFrame>	Data;
	typedef typename Data::value			value; 
	typedef std::function<void(value&)>	Func
	
	public:
	agent(DataFrame& d):data(d); 

	void operator()(); 

	private:
	Data data;
	Func func; 
};
