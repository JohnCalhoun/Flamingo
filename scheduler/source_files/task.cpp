//task.cpp
#include "data.cpp"

class task {	
	typedef std::pair<Input,Output> argument;

	task(); 
	~task(); 


	Input input;
	Output output; 	
	
	std::function<void(argument)> function; 
	
	void in();
	void out();
	void run();

	void add(task);  
}
