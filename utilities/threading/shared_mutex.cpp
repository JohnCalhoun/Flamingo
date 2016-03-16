//shared_mutex.cpp
#ifndef SHARED_MUTEX
#define SHARED_MUTEX

#include <boost/thread.hpp>
#include <mutex>

namespace flamingo {
namespace threading {

struct shared_mutex : public boost::shared_mutex{};

template<typename T>
struct shared_lock_guard : public boost::shared_lock<T> {
	shared_lock_guard(T& lock): boost::shared_lock<T>(lock){};
};

template<typename T>
struct lock_guard : public boost::unique_lock<T> {
	lock_guard(T& lock): boost::unique_lock<T>(lock){};
};

}//threading
}//flamingo 
#endif
