// traits.cpp
#ifndef TRAITS_MEMORY_CPP
#define TRAITS_MEMORY_CPP

#include <type_traits>
namespace std {
	template <typename T>
	struct remove_pointer<Flamingo::Memory::Handle<T> > {
		typedef T type;
	};

}

#endif
