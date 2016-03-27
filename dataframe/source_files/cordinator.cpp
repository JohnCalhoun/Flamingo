//cordinator.cpp
#ifndef DATAFRAME_CORDINATOR
#define DATAFRAME_CORDINATOR

#include <unordered_map>
#include <boost/thread.hpp>
#include <mutex>
#include <tbb/concurrent_hash_map.h>
#include "traits.cpp"

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

	class ARC	{
		typedef std::deque<Key>				LRU_list; 

		enum cases{one,two,three,four}; 
		enum status{t1,b1,t2,b3};

		typedef std::unordered_map<Key,status>	Status_map; 

		typedef tbb::queuing_mutex			Mutex;
		typedef typename Mutex::scoped_lock	lock_guard;

		public:	
		ARC(Cordinator& cor):_cordinator(cor); 	
		~ARC(); 
		void request(Key,Memory); 

		private:
			//key functions used in request
		cases find_case(Key); 
		void unsafe_move(Key,Memory); 
		void arc_body(Key,cases); 
		void remove(Key,cases); 
		Value get_ptr(Key);		
			//functions used in ^
		void change_status(Key,status);	
		void replace();
		void pinned_request(); 
			//helper functions
		void push_front_t1(key);
		void push_front_t2(key); 

		void push_front_b1(key);
		void push_front_b2(key); 

		void pop_back_t1(key);
		void pop_back_t2(key); 

		void pop_back_b1(key);
		void pop_back_b2(key); 
		
		void pop_back_pinned(key);
		void push_front_pinned(key); 
				
		size_t LRU_byte_size(LRU_list&); 
		private:
		Mutex* _mutex_ptr; 

		Cordinator& _cordinator
		size_t	max_device;
		size_t	max_pinned;
		size_t	max_host; 		
		double		p;

		//arc list
		LRU_list		_T1;//on device
		LRU_list		_B1;//in pinned

		LRU_list		_T2;//on device
		LRU_list		_B2;//in pinned

		//LRU list
		LRU_list		_pinned;//lru for pinned memory
	
		//status map, dont search 
		Status_map	_status_map; 
	};
	
	public:
	//constructors
	cordinator():_cache(*this);
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

	void move(Key,Memory,lock_guard&); 
	private:
	//data members
	Map		_map;
	ARC		_cache;
};
#include "arc.inl"
#include "cordinator.inl"
#endif 

