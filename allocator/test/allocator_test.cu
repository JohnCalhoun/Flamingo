// allocator_test.cu
#include <buddy_alloc_p.cpp>
#include <standard_alloc_p.cpp>

#include <location.cu>
#include <gtest/gtest.h>
#include <iostream>

#include <MacroUtilities.cpp>
#include <thread>
#include <iostream>
#include <stdlib.h>

#define ALLOCATOR_THREADS 8

//***************************buddy allocator*****************
template <typename Allocator>
class AllocatorTest : public ::testing::Test {
    protected:
     typedef std::vector<int, Allocator> Vector;

     Vector vector;

     DEFINE(CopyTest, ALLOCATOR_THREADS)
};

template <typename T>
void AllocatorTest<T>::CopyTest() {
     Vector copy_of(vector);
};
#define HOST host
#define MANAGED unified
#define PINNED pinned

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
