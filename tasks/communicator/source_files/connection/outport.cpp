//communicator.cpp
#include <task.cpp>
#include "traits.cpp"
#include <functional>

template<	typename Function,
		typename Messages,
		typename Agents,
		typename Address,
		typename Counts,
		int message_tag>
class outport : public taskBase<Messages,Agents,Counts>{
	private: 
	typedef std::ref_wrapper<Messages> Msg;
	typedef std::ref_wrapper<Agents> Agent_ref;
	typedef std::ref_wrapper<Counts> Counts_ref; 
	typedef std::ref_wrapper<Function> func_ref;
	typedef std::ref_wrapper<Address> address_ref;

	public:
	outport(	Messages& m,
			Agents& a,
			Counts& c
			Function& f,
			Address& ad):
				messages(m),
				agents(a),
				counts(c),
				function(f),
				address(ad); 

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
	address_ref	address;
};
