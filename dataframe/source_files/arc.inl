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
	Memory M)
{
	//assumptions
	//	write access to the key dataframe

	lock_guard guard(*_mutex_ptr); 
	Value dataframe_ptr=get_ptr(key); 
	if(dataframe_ptr->location() != M){
		cases Case=find_case(key); 	
		remove_key(key); 
		switch(M){
			case device:
			{
				arc_body(key,Case);
				unsafe_move(key,device);
				break;
			}
			case pinned:
			{
				pinned_request(key);
				unsafe_move(key,pinned);
				break;
			}
			case host: 
			{
				if(dataframe_ptr->location() == device){
					unsafe_move(key,host);
				}
				break;
			}
			case unified:
			{
				unsafe_move(key,unified);		
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
	typedef typename Status_map::iterator iterator; 
	
	iterator tmp=_status_map.find(key);
	cases Case;

	if(tmp==_status_map.end()){
		status stat=std::get<1>(*tmp); 
		switch(stat){
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
		}//end switch	
	}else{
		Case=four; 
	}//end if
	return Case; 
};

template<typename Object,typename Guard>
void cordinator<Object,Guard>::ARC::remove_key(
			cordinator<Object,Guard>::Key key)
{
	typedef typename LRU_list::iterator list_iterator;

	status stat=std::get<1>(*(_status_map.find(key))); 
	LRU_list* list_ptr;
	switch(stat){
		case t1:	list_ptr=&_T1;
		case t2:	list_ptr=&_T2;
		case b1:	list_ptr=&_B1;
		case b2:	list_ptr=&_B2;
	}	
	list_iterator location=std::find(	list_ptr->begin(),
								list_ptr->end(),
								key);
	list_ptr->erase(location); 	
};


template<typename Object,typename Guard>
void cordinator<Object,Guard>::ARC::unsafe_move(
			cordinator<Object,Guard>::Key key,
			Memory M)
{
	Value ptr=get_ptr(key);
	ptr->unsafe_move(M); 
};

template<typename Object,typename Guard>
void cordinator<Object,Guard>::ARC::arc_body(
			cordinator<Object,Guard>::Key			key,
			cordinator<Object,Guard>::ARC::cases	Case)
{
	switch(Case){
		case one:		
		{

		}
		case two:
		{

		}
		case three:
		{

		}
		case four:
		{
			if(true){
				if(true){
	
				}else{
	
				}
			}
			if(true){
			}
		}
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
void cordinator<Object,Guard>::ARC::change_status(
		cordinator<Object,Guard>::Key			key,
		cordinator<Object,Guard>::ARC::status	stat	)
{
	_status_map.insert(std::make_tuple(key,stat)); 
};

template<typename Object,typename Guard>
void cordinator<Object,Guard>::ARC::replace(
		cordinator<Object,Guard>::ARC::status	stat)
{
	if(true){

	}else{

	}
};

template<typename Object,typename Guard>
void cordinator<Object,Guard>::ARC::pinned_request(
		cordinator<Object,Guard>::Key	key)
{
	typedef typename LRU_list::iterator	list_iterator; 
	typedef std::tuple<Guard,bool>			try_type;

	_pinned.push_front(key); 
	size_t size=LRU_byte_size(_pinned); 
	
	while( size>max_pinned){
		list_iterator	it=_pinned.end()-1;
		bool found=false;
		while(!found){
			Value ptr=get_ptr(*it); 
			try_type result=ptr->try_lock(true); 
			found=std::get<1>(result); 		

			if(found){
				ptr->unsafe_move(host); 
				_pinned.erase(it); 	
			}
			it--; 
		}
		size_t size=LRU_byte_size(_pinned); 
	}
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

#endif			
