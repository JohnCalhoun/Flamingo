// location.inl

template <Region M>
template <typename pointer, typename size_type>
void location<M>::MemCopy(pointer src_ptr, pointer dst_ptr, size_type size) {
     typedef typename std::remove_pointer<pointer>::type value_type;

     typedef thrust::reverse_iterator<pointer> reverse_iterator;
     if (		src_ptr <= (dst_ptr + size/sizeof(value_type)) && 
			dst_ptr <= (src_ptr + size/sizeof(value_type))) {
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
//		cudaPointerAttributes at; 
//		cudaPointerAttributes bt; 
//		cudaPointerGetAttributes(&at,dst_ptr); 
//		cudaPointerGetAttributes(&bt,src_ptr); 
 
	     gpuErrorCheck(cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDefault));
     }
};
template<Region M>
int location<M>::number_of_gpus(){
	int Devices;
	cudaGetDeviceCount(&Devices);
	return Devices; 	
}
template<Region M>
size_t location<M>::max_memory(){
	return ~(0); //virtual memory allows maximum size of 2^64 for unified and host memory 
}
template<>
size_t location<Region::pinned>::max_memory(){
	size_t page_size=sysconf(_SC_PAGE_SIZE);
	size_t pages=sysconf(_SC_PHYS_PAGES);   
	
	return pages*page_size; 
}
template<>
size_t location<Region::device>::max_memory(){
	size_t free;
	size_t total; 
	
	cudaMemGetInfo(&free,&total); 
	return total; //virtual memory allows maximum size of 2^64 for unified and host memory 
}


template<Region M>
size_t location<M>::free_memory(){
	return ~(0); //virtual memory allows maximum size of 2^64 for unified and host memory 
}
template<>
size_t location<Region::pinned>::free_memory(){
	size_t page_size=sysconf(_SC_PAGE_SIZE);
	size_t free_pages=sysconf(_SC_AVPHYS_PAGES);   
	
	return free_pages*page_size; 
}
template<>
size_t location<Region::device>::free_memory(){
	size_t free;
	size_t total; 
	
	cudaMemGetInfo(&free,&total); 
	return free; 
}



template <>
template <typename pointer, typename size_type>
void location<Region::host>::MemCopy(pointer src_ptr, pointer dst_ptr, size_type size) {
     if (src_ptr <= (dst_ptr + size) && dst_ptr <= (src_ptr + size)) {
          std::memmove(dst_ptr, src_ptr, size);
    } else {
          gpuErrorCheck(cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDefault));
     }
};

/** \ingroup allocator-module */
template <Region M>
template <typename pointer, typename Item>
void location<M>::fill_in(pointer dst, int count, Item item) {
     int blocksize;
     int mingridsize;
     int gridsize;

     cudaOccupancyMaxPotentialBlockSize(&mingridsize, &blocksize, (void*)cuda_fill<pointer, Item>, 0, count);
     gridsize = (count + blocksize - 1) / blocksize;

     cuda_fill << <gridsize, blocksize>>> (dst, count, item);
 	gpuErrorCheck(cudaGetLastError() );
     gpuErrorCheck(cudaDeviceSynchronize() );
};
template <>
template <typename pointer, typename Item>
void location<Region::unified>::fill_in(pointer dst, int count, Item item) {
     std::fill_n(dst, count, item);
};
template <>
template <typename pointer, typename Item>
void location<Region::pinned>::fill_in(pointer dst, int count, Item item) {
     std::fill_n(dst, count, item);
};
template <>
template <typename pointer, typename Item>
void location<Region::host>::fill_in(pointer dst, int count, Item item) {
     std::fill_n(dst, count, item);
};

//**************************************HOST***************************
template <>
void* location<Region::host>::New(size_t size) {
     void* p;
	p=std::malloc(size);
//     hostErrorCheck(!p)
     return p;
};
/** \ingroup allocator-module
 */
template <>
void location<Region::host>::Delete(void* p) {
     std::free(p);
};
//**************************************HOST***************************
//**************************************PINNED***************************
template <>
void* location<Region::pinned>::New(size_t size) {
     void* p;
	gpuErrorCheck(	
		cudaMallocHost((void**)&p, size),
		std::bad_alloc	
	);
     return p;
};
template <>
void location<Region::pinned>::Delete(void* p) {
	gpuErrorCheck(	
	     cudaFreeHost(p)
	);	
};
//**************************************PINNED***************************
//**************************************DEVICE***************************
template <>
void* location<Region::device>::New(size_t size) {
     void* p;
	gpuErrorCheck(	
		cudaMalloc((void**)&p, size),
		std::bad_alloc
     );
	return p;
};
template <>
void location<Region::device>::Delete(void* p) {
	gpuErrorCheck(	
		cudaFree(p),
		std::bad_alloc
	);
};
//**************************************DEVICE***************************
//**************************************MANAGED***************************
template <>
void* location<Region::unified>::New(size_t size) {
     void* p;
	gpuErrorCheck(	
		cudaMallocManaged((void**)&p, size),
		std::bad_alloc
     );
	return p;
};

template <>
void location<Region::unified>::Delete(void* p) {
	gpuErrorCheck(	
		cudaFree(p)
	);
};
//**************************************MANAGED***************************
