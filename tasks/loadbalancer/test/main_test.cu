#include <gtest/gtest.h>

#if defined(LOADBALANCER_TEST) || defined(ALL)
	#include "loadbalancer.cu"
#endif

int main(int argc, char **argv) {
     ::testing::InitGoogleTest(&argc, argv);
     return RUN_ALL_TESTS();
};
