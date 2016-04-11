// location.cu
#ifndef LOCATION_POLICY_H
#define LOCATION_POLICY_H

#include <cstdlib>
#include <cstring>
#include <new>
#include <cuda.h>
#include <algorithm>
#include <cuda_occupancy.h>
#include <Handle.cpp>
#include <cstring>
#include <cmath>
#include <type_traits>
#include <thrust/iterator/reverse_iterator.h>
#include <sys/unistd.h>
#include "exceptions.cpp"
#include <iostream>
#include "cuda_functions.cu"

namespace Flamingo{
namespace Memory {

enum class Region { host=0,pinned=1,unified=2,device=3 };

template <Region M>
class location {
    public:
     void* New(size_t);
     void Delete(void*);

     template <typename pointer, typename size_type>
     static void MemCopy(pointer src_ptr, pointer dst_ptr, size_type size);

 	static size_t max_memory(); 
	static size_t free_memory();

	static int number_of_gpus(); 

     template <typename pointer, typename Item>
     void fill_in(pointer dst, int count, Item item);
	
	static const Region memory=M; 
};

#include "location.inl"

}//end Memory
}//end Flamingo

#include "traits.cpp"

#endif
