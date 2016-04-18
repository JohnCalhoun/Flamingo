//exceptions.cpp
#include <exception>
#include <string>
#include <sstream>

namespace Flamingo{
namespace DataFrame{
namespace Exceptions {
class base_exception : public std::exception{}; 

template<class ... Other_Types>
struct host_exception : public base_exception, Other_Types...{
	typedef std::string string; 

	string	file;
	int		line;
	

	host_exception(const string& Fi, int L):
			line(L),file(Fi)	{};

	const char* what()const throw(){
		std::stringstream ss;
		static std::string msg;
		ss <<	" File: " << file << 
				" Line: " << line; 

		msg = ss.str().c_str();
		return msg.c_str();
	}
};
}//end exceptions
}//end Dataframe
}//end Flamingo
