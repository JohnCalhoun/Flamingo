#ifndef ROOT_H
#define ROOT_H
#include<location.cu>
#include<cuda.h>

#define __both__ __device__ __host__ 

template<typename T, typename A, typename L> 
class rootBase{
	public:
	typedef T		value_type;
	//device side
	typedef A		Allocator_dev;

	typedef typename Allocator_dev::Location_Policy	Location_dev;
	typedef typename Allocator_dev::pointer			Pointer_dev;
	typedef typename Allocator_dev::size_type		size_type;
	
	typedef typename Allocator_dev::rebind<Pointer_dev,Location_dev>::other Allocator_ptr_dev;
	typedef typename Allocator_ptr_dev::pointer		Root_dev; 
	//host side
	typedef location<host>						Location_host;
	typedef typename Allocator_dev::rebind<T,Location_host>::other	Allocator_host; 
	
	typedef typename Allocator_host::pointer		Pointer_host;
	typedef typename Allocator_host::rebind<Pointer_host,Location_host>::other 
											Allocator_ptr_host;
	typedef typename Allocator_ptr_host::pointer		Root_host; 
	
	typedef Pointer_dev Pointer; 
	private:
	Allocator_dev allocatorel; 
	size_type _size;

	virtual void	allocate(size_type)=0;
	virtual void	deallocate()=0; 
	virtual void	null()=0; 	
	virtual bool   root()=0;
	public:
	template<typename O>
		void assign(const O&);
	rootBase();
	rootBase(size_type);
	~rootBase(); 

	void clear();
	void resize(size_type);
	__both__
	size_type size()const; 

	__both__ 
	virtual Pointer	get(size_type)const=0;
	virtual void		set(const Pointer&,size_type)=0; 
	void				set(size_type);
};
template<typename T, typename A,typename L>
class rootSingle :public rootBase<T,A,L>{
	private:
	typedef typename rootBase<T,A,L>::Allocator_ptr_host Allocator;
	typedef typename rootBase<T,A,L>::Root_host Root;

	Allocator allocator; 
	Root _root; 
	public:
	typedef typename rootBase<T,A,L>::Pointer Pointer;
	typedef typename rootBase<T,A,L>::size_type size_type;
	//make moveconstructable and copy constructable
	//to make swap work
	rootSingle();
	rootSingle(size_type);
	~rootSingle(); 

	private:
	void			allocate(size_type);
	void			deallocate(); 
	void			null(); 	
	bool			root();
	public:
	__both__ 
	Pointer		get(size_type)const;
	void			set(const Pointer&,size_type); 
	void			sync();
	rootSingle& operator=(const rootSingle&); 
};
template<typename T, typename A, typename L>
class rootDouble :public rootBase<T,A,L>{
	public:
	typedef typename rootBase<T,A,L>::Allocator_ptr_host Allocator_host;
	typedef typename rootBase<T,A,L>::Allocator_ptr_dev Allocator_dev;
	typedef typename rootBase<T,A,L>::Root_host Root_h;
	typedef typename rootBase<T,A,L>::Root_dev Root_d;

	Allocator_host allocatorHost;
	Allocator_dev allocatorDev;
	Root_d	_rootHost;
	Root_h _rootDevice;

	void sync();
	public:
	typedef typename rootBase<T,A,L>::Pointer Pointer;
	typedef typename rootBase<T,A,L>::size_type size_type;
	typedef typename rootBase<T,A,L>::Location_dev Location_dev;

	rootDouble();
	rootDouble(size_type);
	~rootDouble(); 

	private:
	void			allocate(size_type);
	void			deallocate(); 
	void			null();
	bool			root();
	public:
	__both__ 
	Pointer		get(size_type)const;
	void			set(const Pointer&,size_type); 
	rootDouble& operator=(const rootDouble&); 
};

template<typename T,typename A,typename L>
class Root :public rootSingle<T,A,L>{};

template<typename T,typename A>
class Root<T,A,device> :public rootDouble<T,A,device>{};

#undef __both__
#include"root.inl"
#endif
