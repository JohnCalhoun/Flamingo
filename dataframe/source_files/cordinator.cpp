//cordinator.cpp
#ifndef DATAFRAME_CORDINATOR
#define DATAFRAME_CORDINATOR

#include <unordered_map>
#include <boost/thread.hpp>
#include <mutex>
#include <tbb/queuing_mutex.h>
#include <tbb/concurrent_hash_map.h>
#include "traits.cpp"
#include "exceptions.cpp"

namespace Flamingo{
namespace DataFrame{

template<typename Object,typename Guard>
class cordinator {
	public:
	typedef typename traits<int>::size_type			Key;
	typedef Object*							Value;
	typedef Guard								lock_guard; 

	private:
	typedef tbb::concurrent_hash_map<Key,Value>		Map;
	typedef typename Map::const_accessor			r_accessor; 
	typedef typename Map::accessor				w_accessor; 
	typedef typename Map::value_type				value_type; 
	typedef typename Map::iterator				iterator; 

	public:
	class ARC	{
		typedef std::deque<Key>				LRU_list; 

		enum cases			{one,two,three,four}; 
		enum LRU_placement		{t1,b1,t2,b2,NONE};

		typedef std::unordered_map<Key,LRU_placement>	Placement_map; 

		typedef tbb::queuing_mutex			Mutex;
		typedef typename Mutex::scoped_lock	lock_guard;

		typedef std::tuple<Key,Guard>			delete_tuple;
		typedef std::list<delete_tuple>		Lock_list; 

		public:	
		ARC(cordinator& cor):_cordinator(cor),p(0)
		{	_mutex_ptr=new Mutex;
			max_device=Memory::location<Memory::Region::device>::max_memory(); 
			max_pinned=Memory::location<Memory::Region::pinned>::max_memory();
			max_host=Memory::location<Memory::Region::host>::max_memory(); 
		}; 	
		~ARC(){delete _mutex_ptr;}; 
		void request(Key,Memory::Region); 

		private:
			//key functions used in request
		cases	find_case(Key); 
		void		remove_key(Key); 
		void		unsafe_move(Key,Memory::Region); 
		void		arc_body(Key,cases); 
		Value	get_ptr(Key);		
			//functions used in 
		void		change_placement(Key,LRU_placement);	
		void		replace(LRU_placement);
		void		pinned_request(Key); 
			//helper functions
		size_t	LRU_byte_size(LRU_list&); 
		private:
		Mutex* _mutex_ptr; 

		cordinator& _cordinator;
		size_t	max_device;
		size_t	max_pinned;
		size_t	max_host; 	
		size_t	cache_size(){return max_device;};
	
		size_t		p;

		void		push_front_t1(Key);
		void		push_front_t2(Key);
		void		push_front_b1(Key);
		void		push_front_b2(Key);
		void		push_front_NONE(Key); 	
		
		//key remove
		delete_tuple	delete_LRU(LRU_list&); 
		void		unlock_list(Lock_list&); 

		//arc list
		LRU_list		_T1;//on device
		LRU_list		_B1;//in pinned

		LRU_list		_T2;//on device
		LRU_list		_B2;//in pinned

		//LRU list
		LRU_list		_pinned;//lru for pinned memory
	
		//status map, dont search 
		Placement_map	_placement_map; 
	};
	
	public:
	//constructors
	cordinator():_cache(*this){};
	~cordinator();

	//member functions
	private:
	Key objectToKey(Object*); 
	Key insert(Key,Value);  

	public:
	Key insert(Object*);
	void remove(Key); 
	Value find(Key);
	void change(Key,Key);	

	void move(Key,Memory::Region,lock_guard&); 
	private:
	//data members
	Map		_map;
	ARC		_cache;
};
#include "arc.inl"
#include "cordinator.inl"

}//end dataframe
}//end flamingo
#endif 

