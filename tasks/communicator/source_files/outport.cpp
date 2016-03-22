//communicator.cpp
#include <task.cpp>
#include "traits.cpp"
#include <functional>

template<	typename Function,
		typename Messages,
		typename Agents,
		typename Counts,
		int message_tag>
class outport : public taskBase<Messages,Agents,Counts>{
	private: 
	typedef std::ref_wrapper<Messages> Msg;
	typedef std::ref_wrapper<messages> Agent_ref;
	typedef std::ref_wrapper<messages> Counts_ref; 
	typedef std::ref_wrapper<Function> func_ref;

	public:
	outport(	Messages& m,
			Agents& a,
			Counts& c
			Function& f):messages(m),agents(a),counts(c),function(f); 

	private:
	void send();
	void count(); 
	
	public:
	void operator()(){
		send();
		count();
	}; 
	
	private:
	Msg			messages; 
	Agent_ref		agents; 
	Counts_ref	counts; 
	func_ref		function; 
};
