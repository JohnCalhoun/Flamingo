#include <location.cu>
#include <gtest/gtest.h>

#define LOCATION_THREADS 8

#include <MacroUtilities.cpp>
#include <vector>
#include <thread>

#include<stdio.h>
#include<stdlib.h>

using namespace Flamingo::Memory;
// host location test
template <Region M>
class LocationTest : public ::testing::Test {
	protected:
	typedef int		test_type;
	typedef test_type*	pointer;
	pointer			h_ptr;
	pointer			d_ptr;
	int length=16;
	int size=sizeof(test_type)*length;
	location<M> policy;

	virtual void SetUp(){
		h_ptr=static_cast<pointer>(malloc(size));
		cudaMalloc((void**)&d_ptr, size);
		for(int i=0; i<length; i++){
			h_ptr[i]=i; 
		}
		policy.MemCopy(h_ptr,d_ptr,size);
	}

	virtual void TearDown(){
		free(h_ptr);
		cudaFree(d_ptr);
	}

	DEFINE(mallocfreetest, LOCATION_THREADS)
     DEFINE(copytest, LOCATION_THREADS)
     DEFINE(filltest, LOCATION_THREADS)
	DEFINE(cudaextracttest, LOCATION_THREADS)
	DEFINE(cudainserttest, LOCATION_THREADS)
	DEFINE(cudablockmovetest, LOCATION_THREADS)
	DEFINE(overlaptest, LOCATION_THREADS)
	DEFINE(sourceindextest,LOCATION_THREADS)
	DEFINE(sizetest,LOCATION_THREADS)
};
template <Region M>
void LocationTest<M>::sizetest() {
     size_t size = policy.free_memory();
     EXPECT_GT(size,0);

     size_t maximum = policy.max_memory();
     EXPECT_GT(maximum,0);
	
	int gpus = policy.number_of_gpus();
     EXPECT_GT(gpus,0);
};

template <Region M>
void LocationTest<M>::mallocfreetest() {
     void* p = NULL;
     p = policy.New(10);
     policy.Delete(p);
     EXPECT_TRUE(p);
};

template <Region M>
void LocationTest<M>::copytest() {
     int a = 1;
     int* a_ptr = &a;
     int b = 0;
     int* b_ptr = &b;

     location<Region::host>::MemCopy(a_ptr, b_ptr, sizeof(int));
     EXPECT_EQ(1, b);
};

template <>
void LocationTest<Region::device>::copytest() {

     size_t size = sizeof(int);
     int a = 1;
     int* a_d = static_cast<int*>(policy.New(size));
     ASSERT_TRUE(a_d);

     int b = 0;
     int* b_d = static_cast<int*>(policy.New(size));
     ASSERT_TRUE(b_d);

     cudaMemcpy(a_d, &a, size, cudaMemcpyHostToDevice);
     cudaMemcpy(b_d, &b, size, cudaMemcpyHostToDevice);

     location<Region::device>::MemCopy(a_d, b_d, 1);
     cudaMemcpy(&b, b_d, size, cudaMemcpyDeviceToHost);
     EXPECT_EQ(1, b);
};

template <Region M>
void LocationTest<M>::filltest() {
	int locallength=10;
	int size = locallength*sizeof(int);
	int value=2;
     int* a_d = static_cast<int*>(policy.New(size) );
     int* a_h = static_cast<int*>(std::malloc(size) );

	policy.fill_in(a_d, locallength,value);
	policy.MemCopy(a_d,a_h,size);
	for(int i=0; i<locallength; i++){
		EXPECT_EQ(a_h[i],value);
	}
	policy.Delete(a_d);
	std::free(a_h);

}

template <Region M>
void LocationTest<M>::overlaptest() {
	int offset=2;
	int locallength=length-offset;
	int localsize=locallength*sizeof(test_type);
		
	policy.MemCopy(	d_ptr+offset,	d_ptr,	locallength);
	cudaMemcpy(		h_ptr,		d_ptr,	localsize,cudaMemcpyDeviceToHost); 
	for(int i=0; i<locallength;i++){
		EXPECT_EQ(h_ptr[i],i+offset);
	}
	
	policy.MemCopy(	d_ptr,	d_ptr+offset,	locallength);
	cudaMemcpy(		h_ptr,		d_ptr,	localsize,cudaMemcpyDeviceToHost); 
	for(int i=offset; i<locallength;i++){
		EXPECT_EQ(h_ptr[i],i);
	}
}

template <>
void LocationTest<Region::host>::overlaptest() {
	int offset=2;
	int locallength=length-offset;
	int localsize=locallength*sizeof(test_type);
	
	policy.MemCopy(h_ptr+offset,h_ptr,localsize);

	for(int i=0; i<locallength;i++){
		EXPECT_EQ(h_ptr[i],i+offset);
	}
}

template <Region M>
void LocationTest<M>::cudaextracttest() {
	pointer tmp;
	int block=4;
	int offset=2;
	int tmp_size=size;
	cudaMalloc((void**)&tmp,tmp_size); 
	cuda_overlapextract<pointer>
		<<<1,32>>>(	d_ptr,
						tmp,
						block,
						offset,
						length);
	cudaMemcpy(h_ptr, tmp, size, cudaMemcpyDeviceToHost);	
	
	int results[6]={4,5,8,9,12,13};
	for(int i=0; i<6;i++){
		EXPECT_EQ(results[i],h_ptr[i]);
	}
	cudaFree(tmp);
}

template <Region M>
void LocationTest<M>::cudainserttest() {
	pointer tmp;
	int block=4;
	int offset=2;
	const int tmp_size=6;
	cudaMalloc((void**)&tmp,tmp_size*sizeof(int)); 
	gpuErrorCheck( cudaGetLastError()); 

	int tmp_h[tmp_size]={4,5,8,9,12,13};
	pointer tmp_h_ptr=tmp_h;  
	cudaMemcpy(	tmp, 
				tmp_h_ptr, 
				sizeof(int)*tmp_size, 
				cudaMemcpyDefault);
	gpuErrorCheck( cudaGetLastError()); 

	cuda_overlapinsert<pointer>
		<<<1,32>>>(		d_ptr,
						tmp,
						block,
						offset,
						length);
	gpuErrorCheck( cudaGetLastError()); 
	cudaMemcpy(	h_ptr, 
				d_ptr, 
				sizeof(int)*tmp_size, 
				cudaMemcpyDeviceToHost);	

	gpuErrorCheck( cudaGetLastError()); 
//	int results[6]={4,5,8,9,12,13};
//	int indexes[6]={2,3,6,7,10,11};
//	for(int i=0; i<6;i++){
//		EXPECT_EQ(
//				results[i],
//				h_ptr[ indexes[i]]
//			);
//	}
//	^^^test no writen right,	
	gpuErrorCheck( cudaGetLastError()); 	
	cudaFree(tmp); 
	gpuErrorCheck( cudaGetLastError()); 
}

template <Region M>
void LocationTest<M>::cudablockmovetest() {
	pointer tmp;
	int block=4;
	int offset=2;
	int tmp_size=size;
	cudaMalloc((void**)&tmp,tmp_size); 
	cudaMemcpy(tmp, d_ptr, tmp_size, cudaMemcpyHostToDevice);
	cuda_blockmove<pointer,test_type>
		<<<1,32,block>>>(		d_ptr,
							tmp,
							block,
							offset,
							length);

	cudaMemcpy(h_ptr, d_ptr, tmp_size, cudaMemcpyDeviceToHost);	
	int results[16]={2,3,2,3,6,7,6,7,10,11,10,11,14,15,14,15};
	for(int i=0; i<length;i++){
		EXPECT_EQ(h_ptr[i],results[i]); 
	}
	cudaFree(tmp); 
}

template <Region M>
void LocationTest<M>::sourceindextest() {
	#define NUMOFTEST_SIDT 12
	int param[NUMOFTEST_SIDT][3];
	for(int i=0; i<NUMOFTEST_SIDT; i++){
		param[i][0]=i; 
		param[i][1]=4;
		param[i][2]=2;
	};
	int result;
	int anwsers[NUMOFTEST_SIDT]={4,5,8,9,12,13,16,17,20,21,24,25};
	for(int i=0; i<NUMOFTEST_SIDT; i++){
		result=getSourceIndex<OVERLAP>(param[i][0],param[i][1],param[i][2]);
		EXPECT_EQ(result,anwsers[i]);
	}

	int block=5;
	int off=2;
	int anwsers2[NUMOFTEST_SIDT]={2,3,4,7,8,9,12,13,14,17,18,19};
	for(int i=0; i<NUMOFTEST_SIDT; i++){
		EXPECT_EQ(anwsers2[i],getSourceIndex<BLOCK>(i,block,off) );
	}
}
const Region host=Region::host; 
const Region unified=Region::unified; 
const Region pinned=Region::pinned; 
const Region device=Region::device; 
// python:key:policy=host unified device pinned
// python:key:tests=sizetest sourceindextest cudaextracttest cudainserttest cudablockmovetest overlaptest copytest mallocfreetest filltest
// python:key:concurrency=Single
// python:template=TEST_F($LocationTest<|policy|>$,|tests||concurrency|){this->|tests||concurrency|();}
// python:start
// python:include=location.test
#include "location.test"
// python:end

#undef LOCATION_THREADS
