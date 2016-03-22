//chanel.cpp
#include <task.cpp>
#include "../traits.cpp"
#include <functional>

namespace p2p {
template<	typename Message,
		typename Reducer>
class chanel : public taskBase<In,out>{
	typedef std::ref_wrapper<Message> In_msg; 
	typedef std::ref_wrapper<Message> Out_msg; 

	public:
	chanel(In& in,Out& out):inMessages(in),outMessages(out); 

	private:
	void send_recieve();
	void erase_sent();
	void sort();
	void reduce(); 
	
	public:	
	void operator()(){
		translate_address(); 
		send_recieve();
		erase_sent();
		sort();
		reduce(); 	
	};
	
	private: 	
	In_msg		outMessages;
	Out_msg		inMessages;
};

#include "chanel.inl"

}
