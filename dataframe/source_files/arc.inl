//arc.inl
#ifndef ARC_INL
#define ARC_INL

#include "cordinator.cpp"
#include <algorithm> 
#include <numeric>
#include <cstdlib>

template<typename Object,typename Guard>
void cordinator<Object,Guard>::ARC::request(
	cordinator<Object,Guard>::Key key,
	Memory::Region M)
{
	//assume write access to the key dataframe

	lock_guard guard(*_mutex_ptr); 
	Value dataframe_ptr=get_ptr(key); 
	
	if(dataframe_ptr->location() != M){
		cases Case=find_case(key); 	
		remove_key(key); 
		switch(M){
			case Memory::Region::device:
			{
				arc_body(key,Case);
				break;
			}
			case Memory::Region::pinned:
			{
				pinned_request(key);
				break;
			}
			case Memory::Region::host: 
			{
				if(dataframe_ptr->location() != Memory::Region::device){
					unsafe_move(key,Memory::Region::host);
				}
				break;
			}
			case Memory::Region::unified:
			{
				unsafe_move(key,Memory::Region::unified);		
				break;
			}
		}//switch
	}//if
}//request

//--------------------------------------------------
template<typename Object,typename Guard>
cordinator<Object,Guard>::ARC::cases
cordinator<Object,Guard>::ARC::find_case(
			cordinator<Object,Guard>::Key key)
{
	typedef typename Placement_map::iterator iterator; 
	
	iterator tmp=_placement_map.find(key);
	cases Case;

	if(tmp!=_placement_map.end()){
		LRU_placement place=std::get<1>(*tmp); 
		switch(place){
			case t1:
			{	
				Case=one;
				break;
			}
			case t2:
			{
				Case=one;
				break;
			}
			case b1:
			{
				Case=two;
				break;
			}
			case b2:
			{
				Case=three;
				break;
			}
			case NONE:
			{
				Case=four;
				break; 
			}
		}//end switch	
	}else{
		change_placement(key,NONE); 
		Case=four; 
	}//end else if
	return Case; 
};

template<typename Object,typename Guard>
void cordinator<Object,Guard>::ARC::remove_key(
			cordinator<Object,Guard>::Key key)
{
	typedef typename LRU_list::iterator list_iterator;

	LRU_placement place=std::get<1>(*(_placement_map.find(key))); 
	LRU_list* list_ptr=NULL;
	switch(place){
		case t1:	list_ptr=&_T1;
		case t2:	list_ptr=&_T2;
		case b1:	list_ptr=&_B1;
		case b2:	list_ptr=&_B2;
		case NONE: list_ptr=NULL;
	}
	if(list_ptr){	
		list_iterator location=std::find(	list_ptr->begin(),
									list_ptr->end(),
									key);
		if(location!=list_ptr->end()){
			list_ptr->erase(location); 
		}
		location=std::find(	_pinned.begin(),
						_pinned.end(),
						key);
		if( location != _pinned.end() ){
			_pinned.erase(location);	
		}
	}
};


template<typename Object,typename Guard>
void cordinator<Object,Guard>::ARC::unsafe_move(
			cordinator<Object,Guard>::Key key,
			Memory::Region M)
{
	//assumes write access to the key
	Value ptr=get_ptr(key);
	ptr->unsafe_move(M); 
};

template<typename Object,typename Guard>
void cordinator<Object,Guard>::ARC::arc_body(
			cordinator<Object,Guard>::Key			key,
			cordinator<Object,Guard>::ARC::cases	Case)
{
	switch(Case){
		case one:		//already in cache 
		{
			push_front_t2(key); 
		}
		case two:      // was in b1
		{
			p=std::min(	cache_size(),
						p+std::max(	
							LRU_byte_size(_B2)/LRU_byte_size(_B1),
							size_t(1))
					); 
			replace(b1);
			push_front_t2(key); 
		}
		case three:	// was in b2,
		{
			p=std::max(	cache_size(),
						p-std::max(	
							LRU_byte_size(_B1)/LRU_byte_size(_B2),
							size_t(1))
					); 
			replace(b2);
			push_front_t2(key); 
		}
		case four:	// was not tracked 
		{
			size_t request_size=get_ptr(key)->device_size();

			size_t t1_size=LRU_byte_size(_T1); 	
	
			size_t l1_size=t1_size+LRU_byte_size(_B1); 
			size_t l2_size=LRU_byte_size(_T2)+LRU_byte_size(_B2); 

			size_t new_l1_size=l1_size+request_size;
			size_t new_t1_size=t1_size+request_size;

			if( new_l1_size >= cache_size()){ 
				if( new_t1_size > cache_size() ){ 
					Lock_list lock_list; 
					while(new_t1_size > cache_size() ){
						delete_tuple result=delete_LRU(_T1); 
						Key key_tmp=std::get<0>(result); 

						push_front_b1(key_tmp);

						std::get<1>(result)->downgrade_to_reader();
						lock_list.push_back(result); 
						new_t1_size=	LRU_byte_size(_T1)+	
									request_size; 
					}//end while
					unlock_list(lock_list);
				}else{
					Lock_list lock_list; 
					while(new_l1_size >= cache_size() ){
						delete_tuple result=delete_LRU(_B1); 
						Key key_tmp=std::get<0>(result); 

						push_front_NONE(key_tmp);
						unsafe_move(key_tmp,Memory::Region::host);				

						std::get<1>(result)->downgrade_to_reader();
						lock_list.push_back(result);
						new_l1_size=	t1_size+
									LRU_byte_size(_B1)+
									request_size; 
					}//end while
					unlock_list(lock_list);		
					replace(NONE);
				}//end if else
			}else{
				size_t test_size=l1_size+l2_size+request_size;

				if( test_size >= 2*cache_size()){
					Lock_list lock_list; 
					while(  test_size >= 2*cache_size() ){
						delete_tuple result=delete_LRU(_B2); 
						Key key_tmp=std::get<0>(result); 

						push_front_NONE(key_tmp);
						unsafe_move(key_tmp,Memory::Region::host);			

						std::get<1>(result)->downgrade_to_reader();
						lock_list.push_back(result);

						test_size=	l1_size+
									LRU_byte_size(_B2)+
									LRU_byte_size(_T2)+
									request_size; 
					}//end while
					unlock_list(lock_list);
				}//end if	
			}//end if else
			replace(NONE);
			push_front_t1(key); 
		}//end case 4
	}//end switch
};
template<typename Object,typename Guard>
cordinator<Object,Guard>::Value cordinator<Object,Guard>::ARC::get_ptr(
			cordinator<Object,Guard>::Key key)
{
	return _cordinator.find(key); 
};
//-------------------------------------------------
template<typename Object,typename Guard>
void cordinator<Object,Guard>::ARC::change_placement(
		cordinator<Object,Guard>::Key			key,
		cordinator<Object,Guard>::ARC::LRU_placement	place)
{
	_placement_map.insert(std::make_pair(key,place)); 
};

template<typename Object,typename Guard>
void cordinator<Object,Guard>::ARC::replace(
		cordinator<Object,Guard>::ARC::LRU_placement	place)
{

	size_t t2_size=LRU_byte_size(_T2); 	
	size_t t1_size=LRU_byte_size(_T1); 	

	Lock_list lock_list; 
	if(	(t1_size>0) &&
		(	((place==b2) && t2_size>=p) | (t1_size>p) )
	){
		while(t1_size > p ){
			delete_tuple result=delete_LRU(_T1); 
			Key key_tmp=std::get<0>(result); 

			push_front_b1(key_tmp);

			std::get<1>(result)->downgrade_to_reader();				
			lock_list.push_back(result); 
			t1_size=LRU_byte_size(_T1); 
		}
	}else{
		while(t2_size > p ){
			delete_tuple result=delete_LRU(_T2); 
			Key key_tmp=std::get<0>(result); 

			push_front_b1(key_tmp);

			std::get<1>(result)->downgrade_to_reader();
			lock_list.push_back(result); 
			t2_size=LRU_byte_size(_T2); 
		}
	}
	unlock_list(lock_list);
}

template<typename Object,typename Guard>
void cordinator<Object,Guard>::ARC::pinned_request(
		cordinator<Object,Guard>::Key	key)
{
	_pinned.push_front(key); 
	size_t size=LRU_byte_size(_pinned); 
	Lock_list lock_list; 
	
	while( size>max_pinned){
		delete_tuple result=delete_LRU(_pinned); 
		Key key_tmp=std::get<0>(result); 

		change_placement(key_tmp,NONE); 
		unsafe_move(key_tmp,Memory::Region::host);
			
		std::get<1>(result)->downgrade_to_reader();	
		lock_list.push_back(result); 
		size=LRU_byte_size(_pinned); 
	}
	unsafe_move(key,Memory::Region::pinned); 
	unlock_list(lock_list); 
};
//----------------------------------------------------
template<typename Object,typename Guard>
size_t cordinator<Object,Guard>::ARC::LRU_byte_size(		
			cordinator<Object,Guard>::ARC::LRU_list& list)
{
	size_t total=std::accumulate(
		list.begin(),
		list.end(),
		0,
		[this](size_t current,Key key){
			Value ptr=get_ptr(key);
			size_t x=ptr->device_size(); 
			return current+x; 
		}
	);
	return total;
};
//----------------------------------------------
template<typename Object,typename Guard>
void cordinator<Object,Guard>::ARC::push_front_t2(		
			cordinator<Object,Guard>::Key key)
{
	_T2.push_front(key); 
	change_placement(key,t2); 
	unsafe_move(key,Memory::Region::device);
};
template<typename Object,typename Guard>
void cordinator<Object,Guard>::ARC::push_front_t1(		
			cordinator<Object,Guard>::Key key)
{
	_T1.push_front(key); 
	change_placement(key,t1); 
	unsafe_move(key,Memory::Region::device);
};

template<typename Object,typename Guard>
void cordinator<Object,Guard>::ARC::push_front_b1(		
			cordinator<Object,Guard>::Key key)
{
	_B1.push_front(key); 
	change_placement(key,b1); 
	pinned_request(key); 
};
template<typename Object,typename Guard>
void cordinator<Object,Guard>::ARC::push_front_b2(		
			cordinator<Object,Guard>::Key key)
{
	_B2.push_front(key); 
	change_placement(key,b2); 
	pinned_request(key); 
};
template<typename Object,typename Guard>
void cordinator<Object,Guard>::ARC::push_front_NONE(		
			cordinator<Object,Guard>::Key key)
{
	change_placement(key,NONE); 
	unsafe_move(key,Memory::Region::host);
};
//----------------------------------------------
template<typename Object,typename Guard>
	cordinator<Object,Guard>::ARC::delete_tuple
	cordinator<Object,Guard>::ARC::delete_LRU(		
			cordinator<Object,Guard>::ARC::LRU_list& list)
{
	//dont delete first member
	typedef typename LRU_list::reverse_iterator	list_iterator; 

	list_iterator it_start=list.rbegin(); 
	list_iterator it_end=list.rend()+1; 

	bool not_found=true;
	const bool write=true; 
	Key found_key; 	
	Guard guard; 

	while(not_found){
		for(list_iterator it=it_start; it<it_end; it++){
			Value dataframe_ptr=get_ptr(*it); 
			auto result=dataframe_ptr->try_lock(write);
			if(std::get<1>(result)){
				//we aquired the lock
				found_key=*it;
				guard=std::get<0>(result); 
				list.erase(it.base());
				not_found=false;  
				break; 					
			}//end if
		}//end for
	}// end while

	return std::make_tuple(found_key,guard); 
};

template<typename Object,typename Guard>
void cordinator<Object,Guard>::ARC::unlock_list(		
			cordinator<Object,Guard>::ARC::Lock_list& list)
{
	std::for_each(	list.begin(),
				list.end(),
				[this](typename Lock_list::value_type& value){
					Key key_tmp=std::get<0>(value); 
					Value dataframe=get_ptr(key_tmp); 
					dataframe->release(std::get<1>(value));
				}
				);
};













#endif			
