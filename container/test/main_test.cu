#include<gtest/gtest.h>

	#if defined(ROOT_TEST) || defined(ALL) 
	#include"root_test.cu"
	#endif	

	#if defined(TREE_TEST) || defined(ALL) 
	#include"tree_test.cu"
	#endif	

	#if defined(UTILITY_TEST) || defined(ALL) 
	#include"utility_test.cu"
	#endif	

	#if defined(HASHEDARRAYTREE_TEST) || defined(ALL) 
	#include"hashedarraytree_test.cu"
	#endif	

	#if defined(ITERATOR_TEST) || defined(ALL) 
	#include"iterator_test.cu"
	#endif	


int main(int argc, char **argv){
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
};

