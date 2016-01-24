# BuddyAllocator

Components
----------
buddy block allocator
standard allcoator
location policies 

Description
-----------
this project has two goals
1)abstract out the complicated memory model of gpu computing and make it have a more c++ interface.
2)implement a buddy block allocator for gpu memory with a modern c++11 interface
 
the first goal is accomplished by the location policy template class. this provides New and Delete member functions to correspond to a the location. for example
	location<host>::New(size)
will allocate size on the host member and return a pointer to it just as 
	location<pinned>::New(size)
will allocate size on pinned host memory. 

the buddy block allocator allows for more efficient use of gpu memory. gpu memory allocations are significationly more expensive then host allocations. so if a program is repeatedly allocating gpu memory and reuse is possible then the buddy block allocator and speed up memory allocation significantly, 8x to 300x. the buddy allocator is fully compliant with the c++11 allocator concept. 

a standard allocator is also included that acts as a wrapper for the standard new and deletes in the location policy; 

Example
-------
more examples can be seen in the benchmark or test folders.
to allocate storage on the gpu

	buddy_alloc_p<int,location<device> > allocator;
	allocator::pointer device_ptr=allocator.allocate(size);
now the variable device_ptr can be used in any kernel launch as a regular pointer would be; 

Documentation Link
------------------
Doxygen Documentation is being worked on;

Installation,tests, and benchmarks
----------------------------------
The source files not require any configuring or installation but does depend on boost/unordered_map
test require googletest inorder to compile and run.
benchmarks need Celero inorder to compile and run. 

authors
-------
John Calhoun

get involved
------------
if you have any ideas,bugs or suggestions please open an issue!

contact info
------------
john.m.calhoun@ttu.edu


