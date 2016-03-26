//columnbase.cpp
#include"traits.cpp"
#include <array>
struct columnbase{
	virtual ~columnbase(){}; 
};

template<class ... Type>
struct Column_Array {
	typedef std::array<columnbase*,traits<Type...>::_numCol> array; 
};



