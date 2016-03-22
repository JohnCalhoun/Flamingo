//agent.cpp
#include <task.cpp>
#include <functional>

template<class DataFrame>
class agent : public taskBase<DataFrame> {
	private:
	typedef std::ref_wrapper<DataFrame>	Data;
	typedef typename Data::value			value; 
	

	public:
	typedef std::function<void(value&)>	Func
	typedef std::function<void(DataFrame&)>	Init	

	agent(DataFrame& d,Func& f,Init i):data(d),func(f),init(i); 

	void operator()(); 

	private:
	Data data;
	Func func; 
	Init init; 
};

#include "agent.inl"
