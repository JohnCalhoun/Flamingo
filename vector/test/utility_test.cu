#include <location.cu>
#include <gtest/gtest.h>

#include<MacroUtilities.cpp>
#include<vector>
#include<thread>
#include<stdio.h>
#include<HashedUtilites.cu>
#define UTILITY_THREADS 1

template<typename D>
class UtilityTest : public ::testing::Test{
	public: 
	typedef int	type;
	typedef shift_functions<D,type> Util;
	typedef typename Util::cordinate cordinate;
	Util util;
	virtual void SetUp(){
		util.set(2,1);
	}

	DEFINE(emptytest,UTILITY_THREADS)
	DEFINE(SetTest,UTILITY_THREADS)
	DEFINE(NextSizeTest,UTILITY_THREADS)
	DEFINE(NextTest,UTILITY_THREADS)
	DEFINE(MoveTest,UTILITY_THREADS)
};

template<typename D>
void UtilityTest<D>::emptytest(){
};

template<typename D>
void UtilityTest<D>::SetTest(){
	Util local;
	local.set(10,10);
	EXPECT_EQ(local.width,10);
	EXPECT_EQ(local.increment,10);
};

template<>
void UtilityTest<UP>::NextSizeTest(){
	Util local;
	local.set(5,2);
	cordinate cor(2,0);
	int size=local.next_size(cor);
	EXPECT_EQ(size,3);
};
template<>
void UtilityTest<DOWN>::NextSizeTest(){
	Util local;
	local.set(5,6);
	cordinate cor(1,0);
	int size=local.next_size(cor);
	EXPECT_EQ(size,3);
};

template<>
void UtilityTest<DOWN>::NextTest(){
	Util local;
	local.set(5,6);
	cordinate cor(1,0);
	cor=local.next(cor);
	cordinate corR(2,1);
	EXPECT_TRUE(cor==corR);
};
template<>
void UtilityTest<UP>::NextTest(){
	Util local;
	local.set(5,2);
	cordinate cor(2,0);
	cor=local.next(cor);
	cordinate corR(0,0);
	EXPECT_TRUE(cor==corR);
};

template<typename D>
void UtilityTest<D>::MoveTest(){
};

//python:key:tests=MoveTest NextTest emptytest SetTest NextSizeTest
//python:key:direction=DOWN UP
//
//python:template=TEST_F($UtilityTest<|direction|>$,|tests||direction|){this->|tests|();}
//
//python:start
//python:include=utility.test
#include"utility.test"
//python:end

#undef UTILITY_THREADS




