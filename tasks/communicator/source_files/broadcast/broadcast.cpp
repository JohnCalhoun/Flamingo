//broadcast.cpp
#include <task.cpp>
#include "traits.cpp"
#include <functional>

namespace broadcast {
template<	typename Message>
class broadcast : public taskBase<In,out>{
	typedef std::ref_wrapper<Message> In_msg; 
	typedef std::ref_wrapper<Message> Out_msg; 

	public:
	broadcast(In& in,Out& out):inMessages(in),outMessages(out); 

	private:
	void send_recieve();
	void erase_sent();
	void sort();
	void reduce(); 
	
	public:	
	void operator()(){
		send_recieve();
		erase_sent();
	};
	
	private: 	
	In_msg		outMessages;
	Out_msg		inMessages;
};

#include "broadcast.inl"

}//broadcast
