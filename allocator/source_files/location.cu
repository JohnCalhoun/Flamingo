// location.cu
#ifndef LOCATION_POLICY_H
#define LOCATION_POLICY_H

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
/**\struct host
 * \ingroup allocator-module
 *  @brief empty struct representing host side memory
 */
/**\struct pinned
 * \ingroup allocator-module
 *  @brief empty struct representing Cuda pinned memory
 */
/**\struct device
 * \ingroup allocator-module
 *  @brief empty struct representing Cuda raw device memory
 */
/**\struct unified
 * \ingroup allocator-module
 *  @brief empty struct representing Cuda unified memory memory
 */


enum Memory { host,pinned,device,unified };

template <typename pointer, typename Item>
__global__ void cuda_fill(pointer dst, int count, Item item);

template <typename T, typename U>
__global__ void cuda_overlapextract(T, U*, int, int, int);

template <typename T, typename U>
__global__ void cuda_blockmove(T, T, int, int, int);

template <typename T, typename U>
__global__ void cuda_overlapinsert(T, U*, int, int, int);

template <typename pointer, typename value_type>
void cuda_memmove(pointer, pointer, value_type*, int, int, int, int*, int*);

/** \ingroup allocator-module
 *	@brief a class providing a single access point for raw memory allocation
 *
 *	location abstracts the four memory regions in gpu computing and provides
 *a common interface.
 *	Provides New,Delete,and MemCopy functions
 *
 *	\code
 *	//to allocate a pointer of managed memory
 *
 *	location<unified> raw_allocator;
 *	void* p=raw_allocator.New(sizeof(int));
 *	raw_allocator.Delete(p);
 *	\endcode
 *
 *	\param T
 *	one of host,pinned,device or unified to specifiy the location of memory
 *
 *	\todo
 *	provide an interface to streams for the memcopy functions
 *
 * */
template <Memory M>
class location {
    public:
     void* New(size_t);
     void Delete(void*);

     template <typename pointer, typename size_type>
     static void MemCopy(pointer src_ptr, pointer dst_ptr, size_type size);

     template <typename size_type>
     static void MemCopy(Handle_void src_ptr, Handle_void dst_ptr, size_type size);

     template <typename pointer, typename Item>
     void fill_in(pointer dst, int count, Item item);
};

/** \fn void* location<T>::New(size_t s)
*	@brief like new will allocate  memory of size s and return a pointer to
*it
*
*	\code
*	// allocate an integer in unified memory
*
*	location<unified> alloc;
*	void* p=alloc.New(sizeof(int));
*	//---------do stuff---------//
*	alloc.Delete(p);
*	\endcode
 */
/** \fn void location<T>::Delete(void* p)
*	@brief Will deallocate the storage pointed to by p
*
*	undevided behavior if p was allocated by a founction other then
*location<T>::New
*
*	\code
*	// allocate an integer in unified memory
*
*	location<unified> alloc;
*	void* p=alloc.New(sizeof(int));
*	//---------do stuff---------//
*	alloc.Delete(p);
*	\endcode
 */
/** \fn void location<T>::MemCopy<pointer,size_type>(pointer src_ptr, pointer dst_pointer, size_type s)
*	
*	@brief 
*	will copy memory of size s from src_pointer to dst_pointer.
*
*	this static function will work regardless of where the pointers are pointer to in memory
*
*	\code
*	location<host> alloc_host;
*	location<device> alloc_device;
*	void* p=alloc_host.New(sizeof(int));
*	void* q=alloc_device.New(sizeof(int));
*
*	*p=7;
*
*	location<host>.MemCopy(p,q,sizeof(int));
*	// *q now is equal to 7
*	\endcode
*
 */

/** \cond  **/
template <Memory M>
template <typename pointer, typename size_type>
void location<M>::MemCopy(pointer src_ptr, pointer dst_ptr, size_type size) {
     typedef typename std::remove_pointer<pointer>::type value_type;
     typedef thrust::reverse_iterator<pointer> reverse_iterator;
     if (src_ptr <= (dst_ptr + size) && dst_ptr <= (src_ptr + size)) {
          int blocksize[3];
          int mingridsize[3];
          cudaOccupancyMaxPotentialBlockSize(
              mingridsize,
              blocksize,
              (void*)cuda_overlapextract<pointer, value_type>,
              0,
              size);

          cudaOccupancyMaxPotentialBlockSizeVariableSMem(
              mingridsize + 1,
              blocksize + 1,
              (void*)cuda_blockmove<pointer, value_type>,
              [](int blocksize) {
					return sizeof(value_type)*blocksize; },
              size);

          cudaOccupancyMaxPotentialBlockSize(
              mingridsize + 2,
              blocksize + 2,
              (void*)cuda_overlapinsert<pointer, value_type>,
              0,
              size);

          int offset = (src_ptr - dst_ptr);

          int offset_abs = std::abs(offset);
          int block = blocksize[1] + offset_abs;
          int groupsize = block;
          int totalsize = size + offset_abs;

          int numofinserts = (size - 1) / (block);
          int type_size = sizeof(value_type);
          int tmp_size = std::abs(type_size * numofinserts * offset);
          value_type* tmp;
          cudaMalloc((void**)&tmp, tmp_size);

          if (offset > 0) {
               cuda_memmove(src_ptr,
                            dst_ptr,
                            tmp,
                            groupsize,
                            offset,
                            totalsize,
                            mingridsize,
                            blocksize);
          } else {
               reverse_iterator src(src_ptr + size);
               reverse_iterator dst(dst_ptr + size);
               cuda_memmove(src,
                            dst,
                            tmp,
                            groupsize,
                            offset_abs,
                            totalsize,
                            mingridsize,
                            blocksize);
          }
     } else {
          cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDefault);
     }
};
template <>
template <typename pointer, typename size_type>
void location<host>::MemCopy(pointer src_ptr, pointer dst_ptr, size_type size) {
     if (src_ptr <= (dst_ptr + size) && dst_ptr <= (src_ptr + size)) {
          std::memmove(dst_ptr, src_ptr, size);
    } else {
          cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDefault);
     }
};

/** \ingroup allocator-module */
template <Memory M>
template <typename pointer, typename Item>
void location<M>::fill_in(pointer dst, int count, Item item) {
     int blocksize;
     int mingridsize;
     int gridsize;

     cudaOccupancyMaxPotentialBlockSize(&mingridsize, &blocksize, (void*)cuda_fill<pointer, Item>, 0, count);
     gridsize = (count + blocksize - 1) / blocksize;

     cuda_fill << <gridsize, blocksize>>> (dst, count, item);
     cudaDeviceSynchronize();
};
template <>
template <typename pointer, typename Item>
void location<host>::fill_in(pointer dst, int count, Item item) {
     std::fill_n(dst, count, item);
};

//**************************************HOST***************************
template <>
void* location<host>::New(size_t size) {
     void* p;
     p = std::malloc(size);
     if (!p) {
          std::bad_alloc exception;
          throw exception;
     }
     return p;
};
/** \ingroup allocator-module
 */
template <>
void location<host>::Delete(void* p) {
     std::free(p);
};
//**************************************HOST***************************
//**************************************PINNED***************************
template <>
void* location<pinned>::New(size_t size) {
     void* p;
     cudaError_t error = cudaMallocHost((void**)&p, size);
     if (error != cudaSuccess) {
          std::bad_alloc exception;
          throw exception;
     }
     return p;
};
template <>
void location<pinned>::Delete(void* p) {
     cudaFreeHost(p);
};
//**************************************PINNED***************************
//**************************************DEVICE***************************
template <>
void* location<device>::New(size_t size) {
     void* p;
     cudaError_t error = cudaMalloc((void**)&p, size);
     if (error != cudaSuccess) {
          std::bad_alloc exception;
          throw exception;
     }
     return p;
};
template <>
void location<device>::Delete(void* p) {
     cudaFree(p);
};
//**************************************DEVICE***************************
//**************************************MANAGED***************************
template <>
void* location<unified>::New(size_t size) {
     void* p;
     cudaError_t error = cudaMallocManaged((void**)&p, size);
     if (error != cudaSuccess) {
          std::bad_alloc exception;
          throw exception;
     }
     return p;
};

template <>
void location<unified>::Delete(void* p) {
     cudaFree(p);
};
//**************************************MANAGED***************************
//************************************CUDA FUNCTIONS**********************
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
     cuda_overlapextract<pointer, value_type> << <mingridsize[0], blocksize[0]>>> (dst_ptr,
                                                                                   tmp,
                                                                                   groupsize,
                                                                                   offset,
                                                                                   totalsize);
     //block moves
     cudaDeviceSynchronize();
     int SMem = blocksize[1] * sizeof(value_type);
     cuda_blockmove<pointer, value_type> << <mingridsize[0], blocksize[1], SMem>>> (dst_ptr,
                                                                                    dst_ptr,
                                                                                    groupsize,
                                                                                    offset,
                                                                                    totalsize);
     cudaDeviceSynchronize();
     //insert overlaps
     cuda_overlapinsert<pointer, value_type> << <mingridsize[0], blocksize[2]>>> (dst_ptr,
                                                                                  tmp,
                                                                                  groupsize,
                                                                                  offset,
                                                                                  totalsize);
};

//*********************************CUDA FUNCTIONS**********************************
/** \endcond **/

#endif
