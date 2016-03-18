#include <gtest/gtest.h>

#if defined(TASK_TEST) || defined(ALL)
	#include "task.cu"
#endif

#if defined(GRAPH_TEST) || defined(ALL)
	#include "graph.cu"
#endif


int main(int argc, char **argv) {
     ::testing::InitGoogleTest(&argc, argv);
     return RUN_ALL_TESTS();
};
