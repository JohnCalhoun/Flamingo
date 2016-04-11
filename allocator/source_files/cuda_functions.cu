// cuda_functions.cu
#ifndef CUDA_FUNCTIONS_ALLOCATOR_H
#define CUDA_FUNCTIONS_ALLOCATOR_H

#include <cstdlib>
#include <cstring>
#include <new>
#include <cuda.h>
#include <algorithm>
#include <cuda_occupancy.h>
#include <Handle.cpp>
#include <cstring>
#include <cmath>
#include <type_traits>
#include <thrust/iterator/reverse_iterator.h>
#include <sys/unistd.h>
#include "exceptions.cpp"
#include <iostream>

namespace Flamingo{
namespace Memory{

template <typename pointer, typename Item>
__global__ void cuda_fill(pointer dst, int count, Item item) {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     for (int i = idx; i < count; i += gridDim.x * blockDim.x) {
          *(dst + i) = item;
     }
}

enum IndexType {
     OVERLAP,
     BLOCK
};

template <IndexType E>
__device__ __host__ int getSourceIndex(int tid, int block, int offset) {
     return 0;
}
template <>
__device__ __host__ int getSourceIndex<OVERLAP>(int tid, int block, int offset) {
     return block * ((tid) / offset) + (tid % offset) + block;
}
template <>
__device__ __host__ int getSourceIndex<BLOCK>(int tid, int block, int offset) {
     return block * ((tid) / (block - offset)) + (tid % (block - offset)) + offset;
}

template <typename T, typename U>
__global__ void cuda_overlapextract(T src, U* tmp, int block, int off, int count) {
     int bid = blockIdx.x * blockDim.x;
     int tid = bid + threadIdx.x;
     //negative just reverse iterations
     //tmp_index
     int src_index = getSourceIndex<OVERLAP>(tid, block, off);
     int tmp_index = tid;

     int stride_src = blockDim.x * block;
     int stride_dst = blockDim.x * off;

     int i = 0;

     while (src_index < count) {
          tmp[tmp_index] = src[src_index];
          i++;
          tmp_index += stride_dst * i;
          src_index += stride_src * i;
     };
};
template <typename T, typename L>
__global__ void cuda_blockmove(T dst, T src, int block, int off, int count) {
     int bid = blockIdx.x * blockDim.x;
     int tid = bid + threadIdx.x;

     extern __shared__ char share[];
	L* share_ptr=(L*)(&share);

     int src_index;
     int shared_index;
     int dst_index;
     //negatice just revese iterations
     src_index = getSourceIndex<BLOCK>(tid, block, off);
     shared_index = tid;
     dst_index = src_index - off;

     int stride = blockDim.x + off;
     L tmp;
     int i = 0;

     int dst_or = dst_index;
     int shared_or = shared_index;
     while (src_index < count) {
          tmp = src[src_index];
          share_ptr[shared_index] = tmp;
          i++;
          src_index += stride;
          shared_index += blockDim.x;
     };
     __syncthreads();
     i = 0;
     dst_index = dst_or;
     shared_index = shared_or;

     while (dst_index < count) {
          tmp = share_ptr[shared_index];
          dst[dst_index] = tmp;
          i--;
          shared_index += blockDim.x;
          dst_index += stride;
     };
};

template <typename T, typename U>
__global__ void cuda_overlapinsert(T dst, U* tmp, int block, int off, int count) {
     int bid = blockIdx.x * blockDim.x;
     int tid = bid + threadIdx.x;
     //negative just revesre iteration
     int src_index = tid;
     int dst_index = getSourceIndex<OVERLAP>(tid, block, off) - off;

     int stride = blockDim.x * off;

     int i = 0;
     while (dst_index + off < count) {
          dst[dst_index] = tmp[src_index];
          i++;
          src_index += stride * i;
          dst_index += stride * i;
     }
};

template <typename pointer, typename value_type>
void cuda_memmove(pointer src_ptr, pointer dst_ptr, value_type* tmp, int groupsize, int offset, int totalsize, int* mingridsize, int* blocksize) {
	gpuErrorCheck(cudaGetLastError() );
     cuda_overlapextract<pointer, value_type> << <mingridsize[0], blocksize[0]>>> (dst_ptr,
                                                                                   tmp,
                                                                                   groupsize,
                                                                                   offset,
                                                                                   totalsize);

     //block moves
	gpuErrorCheck(cudaGetLastError() );
     gpuErrorCheck(cudaDeviceSynchronize() );
     int SMem = blocksize[1] * sizeof(value_type);
     cuda_blockmove<pointer, value_type> << <mingridsize[0], blocksize[1], SMem>>> (dst_ptr,
                                                                                    dst_ptr,
                                                                                    groupsize,
                                                                                    offset,
                                                                                    totalsize);
	gpuErrorCheck(cudaGetLastError() );
     gpuErrorCheck(cudaDeviceSynchronize() );
	//insert overlaps
     cuda_overlapinsert<pointer, value_type> << <mingridsize[0], blocksize[2]>>> (dst_ptr,
                                                                                  tmp,
                                                                                  groupsize,
                                                                                  offset,
                                                                                  totalsize);
	gpuErrorCheck(cudaGetLastError() );
     gpuErrorCheck(cudaDeviceSynchronize() );
};

}//end Memory
}//end Flamingo
#endif
