// Handle_container_test.cu
#include <Handle.cpp>
#include <Handle_Container.cpp>
#include <gtest/gtest.h>
#include <iostream>

#include "MacroUtilities.cpp"
#include <thread>
#include <iostream>
#include <stdlib.h>

#define HANDLE_CONTAINER_THREADS 8
#define TEST_CONTAINER_SIZE 16

using namespace Flamingo::Memory;

class HandleContainerTest : public ::testing::Test {
    protected:
     virtual void SetUp() {
          for (int i = 0; i < TEST_CONTAINER_SIZE; i++) {
               container.insert(new Handle<int>(4 * i, 4, base));
          }
     }

     typedef Handle<int>* Handle_ptr;

     int base[2];
     Handle_Container<int> container;

     DEFINE(FindHandleTest, HANDLE_CONTAINER_THREADS)
     DEFINE(FindRemoveHandleTest, HANDLE_CONTAINER_THREADS)
     DEFINE(HandleListTest, HANDLE_CONTAINER_THREADS)
     DEFINE(GetRemoveAnyTest, HANDLE_CONTAINER_THREADS)
     DEFINE(PrintTest, HANDLE_CONTAINER_THREADS)
     DEFINE(EmptyTest, HANDLE_CONTAINER_THREADS)
};

void HandleContainerTest::FindHandleTest() {
     Handle_ptr h = container.find_handle(4);
     int offset = h->_offset;
     EXPECT_EQ(4, offset);
}
void HandleContainerTest::FindRemoveHandleTest() {
     std::hash<std::thread::id> hasher;

     std::thread::id this_id = std::this_thread::get_id();
     int id = hasher(this_id);
     id = std::abs(id % TEST_CONTAINER_SIZE) * 4;

     Handle_ptr h = container.find_and_remove_handle(id);
     if (h) {
          int offset = h->_offset;
          EXPECT_EQ(id, offset);
          Handle_ptr h2 = container.find_handle(id);
          EXPECT_FALSE(h2);
     }
}
void HandleContainerTest::HandleListTest() {
     typedef std::vector<Handle_ptr> vector;
     vector vect = container.handle_list();
     int size = vect.size();

     EXPECT_EQ(TEST_CONTAINER_SIZE, size);
}
void HandleContainerTest::GetRemoveAnyTest() {
     Handle_ptr h_ptr = container.get_remove_any();
     EXPECT_FALSE(h_ptr == NULL);
}
void HandleContainerTest::PrintTest() { 
//	std::cout << container << '\n'; 
}

void HandleContainerTest::EmptyTest() { EXPECT_FALSE(container.empty()); }

// python:key:function=FindHandleTest FindRemoveHandleTest HandleListTest
// GetRemoveAnyTest PrintTest EmptyTest
// python:key:concurency=Single Threaded
// python:template=TEST_F(HandleContainerTest,|function||concurency|){this->|function||concurency|();}

// python:start
// python:include=handle_container.test
#include \
    "handle_container.test"
// python:end
// clang-format off
#undef HANDLE_CONTAINER_THREADS
#undef TEST_CONTAINER_SIZE
