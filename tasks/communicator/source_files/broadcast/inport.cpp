//inport.cpp
#include <task.cpp>
#include "../traits.cpp"
#include <functional>

namespace broadcast {
template<	typename Function,
		typename Messages,
		typename Agents,
		int message_tag>
class inport : public taskBase<Messages,Agents>{
	private:
	typedef std::ref_wrapper<Messages> Msg;
	typedef std::ref_wrapper<messages> Agent_ref;
	typedef std::ref_wrapper<Function> func_ref;

	public:
	inport(	Messages& m,
			Agents& a,
			Function& f):messages(m),agents(a),function(f); 

	private:
	void recieve();
	
	public:
	void operator()(){
		recieve();
	}; 
	
	private:
	Msg			messages; 
	Agent_ref		agents; 
	func_ref		function; 
};
#include "inport.inl"

}//broadcast
