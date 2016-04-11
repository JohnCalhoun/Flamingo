#include <location.cu>
#include <gtest/gtest.h>

#include<MacroUtilities.cpp>
#include<allocator.cu>
#include<Tree.cu>
#include<vector>
#include<thread>
#include<stdio.h>
#define TREE_THREADS 1

using namespace Flamingo::Memory;
using namespace Flamingo::Vector;

template<typename T, Region M>
class TreeTest : public ::testing::Test{
	public: 
	typedef standard_alloc_policy<T,location<M> > Allocator; 
	typedef Tree<T,Allocator>	tree; 
	tree globalTree;

	DEFINE(ConstructorTest,	TREE_THREADS)
	DEFINE(GetBranchTest,	TREE_THREADS)
	DEFINE(AddBranchTest,	TREE_THREADS)
	DEFINE(RemoveBranchTest,	TREE_THREADS)
	DEFINE(CopyTest,		TREE_THREADS)
	DEFINE(EqualityTest,	TREE_THREADS)
	DEFINE(AssignmentTest,	TREE_THREADS)
	DEFINE(SetGetTest,		TREE_THREADS)
	DEFINE(SwapTest,		TREE_THREADS)
};
template<typename T,Region M>
void TreeTest<T,M>::AssignmentTest(){
	tree local_tree(2);
	tree local_tree_2(2); 
	local_tree_2=local_tree; 
	EXPECT_TRUE(local_tree_2==local_tree); 

	tree local_tree_3(2);
	tree local_tree_4;
	local_tree_4=local_tree_3; 
	EXPECT_TRUE(local_tree_2==local_tree); 

	tree local_tree_5;
	tree local_tree_6; 
	local_tree_5=local_tree_6;  
	EXPECT_TRUE(local_tree_2==local_tree);


	Tree<T,standard_alloc_policy<T, location<Region::host> > > other(10); 
	tree local_copy;
	other.addbranch(); 
	local_copy=other; 	
	
	Tree<T,standard_alloc_policy<T,location<Region::device> > > other1(10); 
	tree local_copy1;
	other1.addbranch(); 
	local_copy1=other1; 		

	Tree<T,standard_alloc_policy<T,location<Region::pinned> > > other2(10); 
	tree local_copy2;
	other2.addbranch(); 
	local_copy2=other2; 		

	Tree<T,standard_alloc_policy<T,location<Region::unified> > > other3(10); 
	tree local_copy3;
	other3.addbranch(); 
	local_copy3=other3; 		

};

template<typename T,Region M>
void TreeTest<T,M>::ConstructorTest(){
	int target_width=3;
	tree local_tree(target_width); 
	int w=local_tree.width();
	ASSERT_EQ(w,target_width);
};

template<typename T,Region M>
void TreeTest<T,M>::GetBranchTest(){
	tree local_tree2(2);
	local_tree2.addbranch();
	typename tree::pointer p=local_tree2.getbranch(0);
};

template<typename T,Region M>
void TreeTest<T,M>::AddBranchTest(){
	tree local_tree2(2);
	local_tree2.addbranch();
	EXPECT_TRUE(local_tree2.isfree());
	local_tree2.addbranch();
	EXPECT_FALSE(local_tree2.isfree());
};

template<typename T,Region M>
void TreeTest<T,M>::RemoveBranchTest(){	
	tree local_tree(2);
	local_tree.addbranch();
	EXPECT_TRUE(local_tree.isfree());
	local_tree.addbranch();
	EXPECT_FALSE(local_tree.isfree());
	
	local_tree.removebranch();
	EXPECT_TRUE(local_tree.isfree());
};

template<typename T,Region M>
void TreeTest<T,M>::CopyTest(){
	tree tree_1(4); 
	tree tree_2;
	tree_1.addbranch(); 
	tree_2=tree_1;
}

template<typename T, Region M>
void TreeTest<T,M>::EqualityTest(){
	tree tree_1(4); 
	tree tree_2(3);
	tree_1.addbranch(); 
	EXPECT_FALSE(tree_2==tree_1);
	tree_2=tree_1;
	EXPECT_TRUE(tree_2==tree_1); 
};
template<typename T,Region M>
void TreeTest<T,M>::SetGetTest(){
	tree local_tree; 	

	local_tree.resize(7);

	EXPECT_EQ(local_tree.width(),7);

	local_tree.setopenbranch(5);
	EXPECT_EQ(local_tree.openbranch(),5);
};
template<typename T,Region M>
void TreeTest<T,M>::SwapTest(){
	tree tree_1(2);
	tree tree_2(2);
	tree tree_3(2);

	tree_1.addbranch(); 
	tree_1.addbranch();

	tree_2=tree_1;
	tree_2.swap(tree_3);
};

const Region host=Region::host;
const Region pinned=Region::pinned;
const Region device=Region::device;
const Region unified=Region::unified;


//python:key:tests=SwapTest SetGetTest AssignmentTest ConstructorTest GetBranchTest AddBranchTest RemoveBranchTest EqualityTest
//python:key:location=device host pinned unified
//python:key:type=int float
//
//python:template=TEST_F($TreeTest<|type|,|location|>$,|tests||location|){this->|tests|();}
//
//python:start
//python:include=tree.test
#include"tree.test"
//python:end

#undef TREE_THREADS




