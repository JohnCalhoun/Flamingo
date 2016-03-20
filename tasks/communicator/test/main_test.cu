#include <gtest/gtest.h>

#if defined(COMMUNICATOR_TEST) || defined(ALL)
	#include "communicator.cu"
#endif

int main(int argc, char **argv) {
     ::testing::InitGoogleTest(&argc, argv);
     return RUN_ALL_TESTS();
};
