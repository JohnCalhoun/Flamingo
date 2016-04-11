// allocator_test.cu
#include <buddy_alloc_p.cpp>
#include <standard_alloc_p.cpp>
#include <Handle.cpp>

#include <location.cu>
#include <gtest/gtest.h>
#include <iostream>

#include <MacroUtilities.cpp>
#include <thread>
#include <iostream>
#include <stdlib.h>
#include <thrust/device_vector.h>

#define ALLOCATOR_THREADS 8
using namespace Flamingo::Memory;

//***************************buddy allocator*****************
template <typename Allocator>
class AllocatorTest : public ::testing::Test {
    protected:
     typedef std::vector<int, Allocator>	Vector;
	typedef std::vector<int>				Std_vector; 

	Vector vector;
	Std_vector std_vector; 

	AllocatorTest():vector(10),std_vector(10){};

     DEFINE(CopyTest, ALLOCATOR_THREADS)
};

template <typename T>
void AllocatorTest<T>::CopyTest() {

	Vector copy_of(vector);
	thrust::device_vector<int,T> device(std_vector);  
};
#define HOST host
#define MANAGED unified
#define PINNED pinned

const Region host=Region::host;
const Region device=Region::device;
const Region pinned=Region::pinned;
const Region unified=Region::unified;



#define BUDDY buddy_alloc_policy
#define STANDARD standard_alloc_policy
// clang-format off
// python:key:function=CopyTest
// python:key:location=HOST MANAGED PINNED
// python:key:concurency=Single Threaded
/// python:template=TEST_F($AllocatorTest<BUDDY<int,location<|location|> > >$,|function||concurency|){this->|function||concurency|();}

// python:start
// python:include=allocator.test
#include \
    "allocator.test"
// python:end
// clang-format on
#undef ALLOCATOR_THREADS
