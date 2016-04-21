template<typename U>
void Internal::UP::operator()(U& vector){
	std::reverse(vector.begin(),vector.end());
};
template<typename U>
void Internal::DOWN::operator()(U& vector){};

//***************************************************

template<typename D,typename V,typename T,typename L>
void Internal::shift_functions<D,V,T,L>::set(int n){
	increment=n;
};
template<typename D,typename V,typename T,typename L>
int Internal::shift_functions<D,V,T,L>::next_size(Internal::shift_functions<D,V,T,L>::cordinate p){
	int x=p.offset();
	int a=p.width()-x;
	int b=p.width()-op(x,increment)%p.width();
	return std::min(a,b); 
};
template<typename D,typename V,typename T,typename L>
Internal::shift_functions<D,V,T,L>::cordinate 
	Internal::shift_functions<D,V,T,L>::next(
		Internal::shift_functions<D,V,T,L>::cordinate p)
{
	return p+=next_size(p); 
};
template<typename D,typename V,typename T,typename L>
void Internal::shift_functions<D,V,T,L>::adjust(V& vector){
	direction(vector);
};

template<typename pointer,unsigned int blockSize>
__global__ void Internal::tree_leave_equality(	pointer p_1,
										pointer p_2,
										bool* result, 
										int size){
	extern __shared__ bool sdata[]; 
	
	//load shared data
	unsigned int tid=threadIdx.x;
	unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int gridSize=blockSize*2*gridDim.x; 
	sdata[tid]=true;
	
	while(i<size){
		bool comp_1=(p_1[i]==p_2[i]); 
		bool comp_2=(p_1[i+blockDim.x]==p_2[i+blockDim.x]); 
		sdata[tid]=(comp_1 && comp_2 ); 
		i+=gridSize; 
	}
	__syncthreads(); 

	//reduce in shared mem
	if(blockSize>=512){
		if(tid<256){
			sdata[tid]&=sdata[tid+256];
		}
		__syncthreads();
	}
	if(blockSize>=256){
		if(tid<128){
			sdata[tid]&=sdata[tid+128];
		}
		__syncthreads();
	}
	if(blockSize>=128){
		if(tid<64){
			sdata[tid]&=sdata[tid+64];
		}
		__syncthreads();
	}

	if(tid<32){
		if(blockSize>=64)sdata[tid]&=sdata[tid+32];
		if(blockSize>=32)sdata[tid]&=sdata[tid+16];
		if(blockSize>=16)sdata[tid]&=sdata[tid+8];
		if(blockSize>=8)sdata[tid]&=sdata[tid+4];
		if(blockSize>=4)sdata[tid]&=sdata[tid+2];
		if(blockSize>=2)sdata[tid]&=sdata[tid+1];
	}
	//write result to global mem
	if (tid==0) result[blockIdx.x]=sdata[0];  
	
};
/************************************equality operator******************/
template<typename A,typename B,typename C,typename D>
bool Internal::Equality_false<A,B,C,D>::operator()(	const Tree<A,B>& tree_1, 
									const Tree<C,D>& tree_2){
	return false; 
};
template<typename T,typename L>
bool Internal::Equality_device<T,L>::operator()(	const Tree<T,L>& tree_1, 
									const Tree<T,L>& tree_2){
	int width=tree_1.width();
	typedef typename Tree<T,L>::pointer	pointer;
	typedef Memory::location<Memory::Region::device>		Location;
	Location location;

	if(tree_1.width()==tree_2.width()){	
		int grid=tree_1._cudaMingridSize;
		int block=tree_1._cudaBlockSize; 	
		bool result=true; 
		
		bool* result_temp=static_cast<bool*>(location.New(sizeof(bool)*grid)); 
		bool result_a[grid];
		int width=tree_1.width();
		for(int i=0; i<width; i++){
			
			pointer ptr_1=tree_1.getbranch(i);
			pointer ptr_2=tree_2.getbranch(i); 
			#define kernel(x) 	case x: \
							Internal::tree_leave_equality<pointer,x> \
							<<<grid,block,block>>>(ptr_1,ptr_2,result_temp,width); \
							break; 
			if(ptr_1 && ptr_2){
				
				switch(block){ 
					kernel(512)
					kernel(256)
					kernel(128)
					kernel(64)
					kernel(32)
					kernel(16)
					kernel(8)
					kernel(4)
					kernel(2)
					kernel(1)
				}
				#undef kernel
				cudaDeviceSynchronize(); 
				location.MemCopy(result_temp,result_a,sizeof(bool)*grid);
				result=std::any_of(result_a,result_a+grid, [](bool b){return ~b;} );
				
			}else{	
				bool p_1=ptr_1;
				bool p_2=ptr_2;
				if(p_1 xor p_2)
					result=false;
				else
					result=true;
				
			};
			if(!result)
				break;
		}
		location.Delete(result_temp); 
		return result;	
	}else{
		return false;	
	}
};
template<typename T,typename L>
bool Internal::Equality_host<T,L>::operator()(	const Tree<T,L>& tree_1, 
									const Tree<T,L>& tree_2){
	typedef typename Tree<T,L>::pointer pointer;

	if(tree_1.width()==tree_2.width()){
		int width=tree_2.width();
		bool result=true; 
		for(int i=0; i<width; i++){
			pointer ptr_1=tree_1.getbranch(i);
			pointer ptr_2=tree_2.getbranch(i); 
			if(ptr_1 && ptr_2){
				result=std::equal(	ptr_1,
								ptr_1+width,
								ptr_2
								); 
			}else{
				bool p_1=ptr_1;
				bool p_2=ptr_2;
				if(p_1 xor p_2)
					result=false;
				else
					result=true;
			}
			if(!result)
				break;
		}
		return result;
	}else{
		return false;	
	}
};







