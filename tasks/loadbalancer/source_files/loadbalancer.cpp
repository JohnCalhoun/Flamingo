//loadbalancer.cpp
#include <task.cpp>
#include <functional.cpp> 

class loadbalancer_base{
	static std::size_type		total; 
	static std::mutex			lock;

	void add(std::size_type); 
	void subtract(std::size_type); 
};
std::size_type loadbalancer_base::total=0;
std::mutex	loadbalancer_base::lock; 

template<typename Agents,typename Counts>
class loadbalancer : public taskBase<Agents,Counts>,loadbalancer_base{
	private
	typedef std::ref_wrapper<Agents> agent_ref;
	typedef std::ref_wrapper<Counts> counts_ref; 

	public:
	loadbalancer(Agents& a,Counts& c):agent(a),counts(c); 

	private:
	void update(); 
	void move();

	public: 
	void operator()(){
		update();
		move(); 
	}; 

	private:	
	agents_ref agent;
	counts_ref counts; 
};
