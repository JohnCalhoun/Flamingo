// free_list_test.cu
#include <Free_List.cpp>
#include <Handle.cpp>

#include <mutex>
#include <gtest/gtest.h>

#include <iostream>
#include <MacroUtilities.cpp>
#include <thread>
#include <iostream>
#include <stdlib.h>

#define FREE_LIST_THREADS 8
#define TEST_CONTAINER_SIZE 16
#define TEST_SIZE 4

using namespace Flamingo::Memory;

class FreeListTest : public ::testing::Test {
    public:
     typedef Handle<int> handle;
     typedef handle* handle_ptr;
     int* base;

    protected:
     virtual void SetUp() {
          base = new int[TEST_CONTAINER_SIZE];
          for (int j = 0; j < TEST_CONTAINER_SIZE / 2; j++) {
               free_list.add_free_handle(new handle(2 * j * TEST_SIZE, TEST_SIZE, base));
               free_list.add_reserved_handle(
                   new handle((2 * j + 1) * TEST_SIZE, TEST_SIZE, base));
          }
     }
     virtual void TearDown() {
          delete base;
     };
     Free_List<int> free_list;
     int id2off();

     DEFINE(CoutTest, FREE_LIST_THREADS)
     DEFINE(FindFreeHandleTest, FREE_LIST_THREADS)
     DEFINE(FindReservedHandleTest, FREE_LIST_THREADS)
     DEFINE(SplitTest, FREE_LIST_THREADS)
     DEFINE(CombineTest, FREE_LIST_THREADS)
     DEFINE(VacantTest, FREE_LIST_THREADS)
};
int FreeListTest::id2off() {
     std::hash<std::thread::id> hasher;

     std::thread::id this_id = std::this_thread::get_id();
     int id = hasher(this_id);
     id = std::abs(id % TEST_CONTAINER_SIZE) * TEST_SIZE / 2;
     return id;
}

void FreeListTest::CoutTest() { 
//std::cout << free_list << '\n'; 
}
void FreeListTest::FindFreeHandleTest() {

     auto h = free_list.find_free_handle(TEST_SIZE);
     ASSERT_TRUE(h);
     EXPECT_EQ(h->_size, TEST_SIZE);
}
void FreeListTest::FindReservedHandleTest() {
     int id = id2off();
     id = 2 * id + 1;

     auto h = free_list.find_reserved_handle(id);
     if (h) {
          EXPECT_EQ(h->_size, TEST_SIZE);
          EXPECT_EQ(h->_offset, id);
     }
}
void FreeListTest::SplitTest() {

     auto h = free_list.find_free_handle(TEST_SIZE);
     ASSERT_TRUE(h);
     int offset = h->_offset;
     free_list.split(h, 2);
     EXPECT_EQ(h->_size, 2);
     EXPECT_EQ(h->_offset, offset);
}
void FreeListTest::CombineTest() {
     int id = id2off();
     id = 2 * id + 1;

     auto h = free_list.find_reserved_handle(id);
     if (h) {
          free_list.combine(h);
          ASSERT_TRUE(h);
          EXPECT_EQ(h->_offset, 0);
          EXPECT_EQ(h->_size, 8);
     }
}

void FreeListTest::VacantTest() {
     EXPECT_TRUE(free_list.vacant());
};
// clang-format off
// python:key:function=CoutTest FindFreeHandleTest FindReservedHandleTest
// SplitTest CombineTest VacantTest
// python:key:concurency=Single Threaded
// python:template=TEST_F(FreeListTest,|function||concurency|){this->|function||concurency|();}

// python:start
// python:include=free_list.test
#include \
    "free_list.test"
// python:end
// clang-format on
#undef FREE_LIST_THREADS
#undef TEST_CONTAINER_SIZE
#undef TEST_SIZE
