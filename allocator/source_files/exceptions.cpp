//exceptions.cpp
#ifndef ALLOCATION_EXCEPTIONS_CPP
#define ALLOCATION_EXCEPTIONS_CPP

#include <exception>
#include <string>
#include <sstream>


class base_exception : public std::exception{}; 

template<class ... Other_Types>
struct host_exception : public base_exception, Other_Types...{
	typedef std::string string; 

	string	file;
	int		line;
	

	host_exception(string& Fi, int L):
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

template<class ... Other_types>
struct cuda_exception : public base_exception, Other_types...{	
	typedef std::string string; 

	string	error_description;
	string	error_name;
	string	file;
	int line;
	

	cuda_exception(	cudaError_t  error,
					string Fi,int L):
			line(L),file(Fi)
	{
		error_description=cudaGetErrorString(error); 
		error_name=cudaGetErrorName(error); 
	}

	const char* what()const throw(){
		std::stringstream ss;
		static std::string msg;
		ss <<	" File: " << file << 
				" Line: " << line << 
				" Error: " << error_name <<
				" Description: "<< error_description; 
		msg = ss.str().c_str();
		return msg.c_str();
	}
};

template<class ... Tags>
void raise_cuda_exception(	cudaError_t error,
						std::string file, 
						int line){
	if(error != cudaSuccess){
		cuda_exception<Tags...> ex(error,file,line);
		throw ex; 
	}	
}

template<class ... Tags>
void raise_host_exception(	bool error, 
						std::string file, 
						int line){
	if(error){
		host_exception<Tags...> ex(file,line);
		throw ex; 
	}	
}

//could add further file and line additions 
//#ifndef RELEASE
//#define gpuErrorCheck(ans,Types...){  raise_cuda_exception<Types>(ans, __FILE__, __LINE__);} 
//#define hostErrorCheck(ans,Types...){ raise_host_exception<Types>(ans, __FILE__, __LINE__);}
//#else
#define gpuErrorCheck(ans,Types...){ans;} 
#define hostErrorCheck(ans,Types...){ans;}
//#endif

#endif


