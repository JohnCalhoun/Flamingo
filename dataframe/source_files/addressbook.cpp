//traits.cpp
#ifndef DATAFRAME_BOOK
#define DATAFRAME_BOOK

#include <map>
#include <boost/thread.hpp>

template<typename T,typename P>
class addressbook : public std::map<T,P>, public boost::shared_mutex
{};

#endif 

