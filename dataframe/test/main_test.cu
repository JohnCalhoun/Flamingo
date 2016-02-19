#include <gtest/gtest.h>

#if defined(DATAFRAME_TEST) || defined(ALL)
#include "dataframe_test.cu"
#endif

int main(int argc, char **argv) {
     ::testing::InitGoogleTest(&argc, argv);
     return RUN_ALL_TESTS();
};
