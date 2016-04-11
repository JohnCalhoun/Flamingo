// allocation_test.cu
#include <buddy_alloc_p.cpp>
#include <standard_alloc_p.cpp>

#include <location.cu>

#include <gtest/gtest.h>
#include <iostream>

#include <MacroUtilities.cpp>
#include <thread>
#include <iostream>
#include <stdlib.h>

#define ALLOCATION_THREADS 8
#define TEST_CONTAINER_SIZE 16
#define TEST_SIZE 4

using namespace Flamingo::Memory; 

//***************************buddy allocator*****************
template <typename Policy>
class BuddyAllocTest : public ::testing::Test {
    protected:
     Policy allocator;

     DEFINE(MaxSizeTest, ALLOCATION_THREADS)
     DEFINE(AllocDeallocTest, ALLOCATION_THREADS)
     DEFINE(MultiAllocDeallocTest, ALLOCATION_THREADS)
     DEFINE(VectorAllocDeallocTest, ALLOCATION_THREADS)
     DEFINE(RebindTest, ALLOCATION_THREADS)
};

template <typename T>
void BuddyAllocTest<T>::MaxSizeTest() {
     auto p = (this->allocator).max_size();
}

template <typename T>
void BuddyAllocTest<T>::AllocDeallocTest() {
     auto q = (this->allocator).allocate(sizeof(int));
     (this->allocator).deallocate(q);

     q = (this->allocator).allocate(1);
	(this->allocator).deallocate(q);
}

template <typename T>
void BuddyAllocTest<T>::MultiAllocDeallocTest() {
     auto q1 = (this->allocator).allocate(sizeof(int));
     auto q2 = (this->allocator).allocate(sizeof(int));

     (this->allocator).deallocate(q1);
     (this->allocator).deallocate(q2);

	for(int i=1; i<20; i++){
		q1 = (this->allocator).allocate(i*sizeof(int));
		q2 = (this->allocator).allocate(i*sizeof(int));

		int* p1=q1;
		int* p2=q2;	

		T::Location_Policy::MemCopy(p1,p2,i*sizeof(int) );
		
		(this->allocator).deallocate(q1);
		(this->allocator).deallocate(q2);
	}


}

template <typename T>
void BuddyAllocTest<T>::VectorAllocDeallocTest() {
     typedef typename T::pointer point;
     std::vector<point> pointers(20);

     for (int i = 0; i < 19; i++) {
          pointers[i] = (this->allocator).allocate(sizeof(int));
     }
     for (int i = 0; i < 19; i++) {
          (this->allocator).deallocate(pointers[i]);
     }
}

template <typename T>
void BuddyAllocTest<T>::RebindTest() {
     typedef typename T::rebind<float, location<Region::host> >::other other;
     other alloc_other;
     typedef typename other::pointer other_pointer;
     other_pointer p;
     p = alloc_other.allocate(sizeof(float));
     alloc_other.deallocate(p);

     typedef typename T::pointer pointer;
     typedef typename T::rebind<pointer, location<Region::host> >::other ptr_Allocator;
     ptr_Allocator ptr_alloc;
     typename ptr_Allocator::pointer q = ptr_alloc.allocate(0);
     ptr_alloc.deallocate(q);
}

#define HOST host
#define DEVICE device
#define MANAGED unified
#define PINNED pinned

#define BUDDY buddy_alloc_policy
#define STANDARD standard_alloc_policy

const Region host=Region::host;
const Region unified=Region::unified;
const Region pinned=Region::pinned;
const Region device=Region::device;

// clang-format off
// python:key:function=MaxSizeTest AllocDeallocTest MultiAllocDeallocTest
// VectorAllocDeallocTest RebindTest
// python:key:location=HOST DEVICE MANAGED PINNED
// python:key:policy=BUDDY STANDARD
// python:key:concurency=Single Threaded
// python:template=TEST_F($BuddyAllocTest<|policy|<int,location<|location|> > >$,|function||concurency|){this->|function||concurency|();}

// python:start
// python:include=allocation.test
#include \
    "allocation.test"
// python:end
// clang-format on
#undef ALLOCATION_THREADS
#undef TEST_CONTAINER_SIZE
#undef TEST_SIZE
