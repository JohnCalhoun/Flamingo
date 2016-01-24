#ifndef MACRO_UTILITIES_H
#define MACRO_UTILITIES_H

#include<vector>
#include<thread>

#define DEFINE(name,threads)	\
	void name();			\
	REGISTER(name,threads)	\


#define REGISTER(name,threads)		\
	SINGLE(name)					\
	THREADED(name,threads)			\

#define THREADED(name, num_of_threads)			\
	void name##Threaded(){					\
		std::vector< std::thread > threads;		\
		for(int i=0; i<num_of_threads; i++){		\
			threads.push_back( std::thread([this](){this->name##Single();}));	\
		};									\
		for(auto &t : threads){					\
			t.join();							\
		};									\
	};

#define SINGLE(name)			\
	void name##Single(){		\
		this->name();				\
	};




#endif
