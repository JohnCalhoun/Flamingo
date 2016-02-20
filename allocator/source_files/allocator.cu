// allocator_policy.cpp

#ifndef ALLOCATOR_POLICY_CPP
#define ALLOCATOR_POLICY_CPP

#include <buddy_alloc_p.cpp>
#include <standard_alloc_p.cpp>
#include <location.cu>
/** \ingroup allocator-module
 * @brief
 * Just a wrapper class to make a single interface to the allocator types
 *
 * for locations device,pinned,managed the implementation is a buddy block
 *allocator.
 * host location the default is the standard allocator.
 * @param T
 * type being allocated by allocator
 * @param L
 * location policy. one of host,device,pinned,managed
 * if L is host and T is not given then T defaults to standard_alloc_p
 *
 * \code
 * //to insitiate a allocater with buddy block strategy on the device and then
 *allocate
 *
 *
 * typedef allocation_policy<int,device>::allocator Allocator;
 * Allocator allocator;
 * Allocator::pointer p=allocator.allocate(sizeof(int)*2);
 * \endcode
 *
 */
template <typename T, Memory M>
struct allocation_policy {
     typedef buddy_alloc_policy<T, location<M> > allocator;
};

template <typename T>
struct allocation_policy<T, host> {
     typedef standard_alloc_policy<T, location<host> > allocator;
};

#endif
