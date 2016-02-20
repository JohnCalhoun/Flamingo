#include <gtest/gtest.h>

#if defined(DATAFRAME_TEST) || defined(ALL)
	#include "container.cu"
#endif

#if defined(ITERATOR_TEST) || defined(ALL)
	#include "iterator.cu"
#endif


int main(int argc, char **argv) {
     ::testing::InitGoogleTest(&argc, argv);
     return RUN_ALL_TESTS();
};
