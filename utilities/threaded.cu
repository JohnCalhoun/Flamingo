#ifndef THREADED_H
#define THREADED_H

#include<vector>
#include<thread>

#define THREADED(class,function, num_of_threads)		\
	void class::function##Threaded(){				\
		std::vector< std::thread > threads;		\
		for(int i=0; i<num_of_threads; i++){		\
			threads.push_back( std::thread([this](){this->function##Single();}));	\
		};									\
		for(auto &t : threads){					\
			t.join();							\
		};									\
	};

#define THREADED_T(class,function, num_of_threads)	\
	template<typename T>						\
	void class<T>::function##Threaded(){			\
		std::vector< std::thread > threads;		\
		for(int i=0; i<num_of_threads; i++){		\
			threads.push_back( std::thread(		\
					[this](){this->function##Single();}	\
					));						\
		}									\
		for(auto &t: threads){					\
			t.join();							\
		}									\
	}

#endif
