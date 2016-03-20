#include <gtest/gtest.h>

#if defined(AGENT_TEST) || defined(ALL)
	#include "agent.cu"
#endif

int main(int argc, char **argv) {
     ::testing::InitGoogleTest(&argc, argv);
     return RUN_ALL_TESTS();
};
