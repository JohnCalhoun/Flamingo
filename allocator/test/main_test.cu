#include <gtest/gtest.h>

#if defined(LOCATION_TEST) || defined(ALL)
#include \
    "location_test.cu"
#endif

#if defined(HANDLE_TEST) || defined(ALL)
#include \
    "handle_test.cu"
#endif

#if defined(HANDLE_C_TEST) || defined(ALL)
#include \
    "handle_container_test.cu"
#endif

#if defined(FREE_TEST) || defined(ALL)
#include \
    "free_container_test.cu"
#endif

#if defined(FREE_LIST_TEST) || defined(ALL)
#include \
    "free_list_test.cu"
#endif

#if defined(ALLOCATION_TEST) || defined(ALL)
#include \
    "allocation_test.cu"
#endif

#if defined(ALLOCATOR_TEST) || defined(ALL)
#include \
    "allocator_test.cu"
#endif

int main(int argc, char **argv) {
     ::testing::InitGoogleTest(&argc, argv);
     return RUN_ALL_TESTS();
};
