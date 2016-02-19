// free_list_test.cu
#include <dataframe.cpp>
#include <gtest/gtest.h>

#include "MacroUtilities.cpp"

#include <cstddef>
#include <type_traits>
#define DATAFRAME_THREADS 8

class DataFrameTest : public ::testing::Test {
    protected:
     virtual void SetUp() {}

     DEFINE(EmptyTest, DATAFRAME_THREADS)
};
void DataFrameTest::EmptyTest() {
};
// python:key:function=EmptyTest
// python:key:concurency=Single Threaded
// python:template=TEST_F(DataFrameTest,|function||concurency|){this->|function||concurency|();};
// python:start
// python:include=dataframe.test
#include "dataframe.test"
// python:end
#undef DataFrame_THREADS
