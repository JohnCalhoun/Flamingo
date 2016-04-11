// allocator_policy.cpp

#ifndef ALLOCATOR_POLICY_CPP
#define ALLOCATOR_POLICY_CPP

#include <buddy_alloc_p.cpp>
#include <standard_alloc_p.cpp>
#include <location.cu>

namespace Flamingo{
namespace Memory{

template <typename T, Region M>
struct allocation_policy {
     typedef buddy_alloc_policy<T, location<M> > allocator;
};

template <typename T>
struct allocation_policy<T, Region::host> {
     typedef standard_alloc_policy<T, location<Region::host> > allocator;
};

}//end Memory
}//end Flamingo

#endif
