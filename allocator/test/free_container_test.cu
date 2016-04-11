// free_container_test.cu
#include <Handle.cpp>
#include <Free_Container.cpp>
#include <gtest/gtest.h>

#include <iostream>
#include <thread>
#include <iostream>
#include <stdlib.h>

#include <MacroUtilities.cpp>

#define TEST_CONTAINER_SIZE 16
#define TEST_SIZE 4

using namespace Flamingo::Memory;

class FreeContainerTest : public ::testing::Test {
    public:
     typedef Handle<int>* Handle_ptr;

    protected:
     virtual void SetUp() {
          for (int i = 0; i < TEST_CONTAINER_SIZE; i++) {
               Handle<int>* handle_ptr = new Handle<int>(TEST_SIZE * i, TEST_SIZE, base);
               container.insert(handle_ptr);
          }
     }
     virtual void TearDown() {
     }
     int base[2];
     Free_Container<int> container;
     int get_id();

#define FREE_CONTAINER_THREADS 8
     DEFINE(FindReturnHandleTest, FREE_CONTAINER_THREADS)
     DEFINE(FindRemoveHandleTest, FREE_CONTAINER_THREADS)
     DEFINE(Size2OrderTest, FREE_CONTAINER_THREADS)
     DEFINE(HandleListTest, FREE_CONTAINER_THREADS)
     DEFINE(CoutTest, FREE_CONTAINER_THREADS)
     DEFINE(EmptyTest, FREE_CONTAINER_THREADS)
#undef FREE_CONTAINER_THREADS
};

int FreeContainerTest::get_id() {
     std::hash<std::thread::id> hasher;

     std::thread::id this_id = std::this_thread::get_id();
     int id = hasher(this_id);
     id = std::abs(id % TEST_CONTAINER_SIZE) * TEST_SIZE;
     return id;
}

void FreeContainerTest::FindReturnHandleTest() {
     int id = get_id();
     id = 0;
     Handle_ptr h = container.find_return_handle(TEST_SIZE, id);
     if (h) {
          ASSERT_TRUE(h);
          int size = h->_size;
          EXPECT_EQ(TEST_SIZE, size);
     }
}
void FreeContainerTest::FindRemoveHandleTest() {
     int id = get_id();
     id = 0;
     Handle_ptr h = container.find_remove_handle(TEST_SIZE);
     ASSERT_TRUE(h);
     bool found = container.find_remove_handle(TEST_SIZE, id);
     ASSERT_FALSE(found);
}

void FreeContainerTest::Size2OrderTest() {
     int order = container.size2order(4);
     EXPECT_EQ(2, order);
}
void FreeContainerTest::HandleListTest() {
     std::vector<Handle<int>*> vec;
     vec = (container.handle_list());
}
void FreeContainerTest::CoutTest() { 
//std::cout << container << '\n'; 
}
void FreeContainerTest::EmptyTest() {
     Free_Container<int> empty_container;
     ASSERT_TRUE(empty_container.empty());
     ASSERT_FALSE(container.empty());
}
// clang-format off
// python:key:function=FindReturnHandleTest FindRemoveHandleTest Size2OrderTest
// HandleListTest CoutTest EmptyTest
// python:key:concurency=Single Threaded
// python:template=TEST_F(FreeContainerTest,|function||concurency|){this->|function||concurency|();}
//
// python:start
// python:include=free_container.test
#include \
    "free_container.test"
// python:end
// clang-format on
#undef FREE_CONTAINTER_TEST
#undef TEST_SIZE
