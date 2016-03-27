//arc.inl
#ifndef ARC_INL
#define ARC_INL

#include "cordinator.cpp"

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
		remove(key,Case); 

		switch(M){
			case device:
			{
				arc_body(key,Case);
				unsafe_move(key,device);
			}
			case pinned:
			{
				//run pinned LRU algorithm 
				unsafe_move(key,pinned);
			}
			case host: 
			{
				if(dataframe_ptr->location() == device){
					unsafe_move(key,host);
				}
			}
			case unified:
			{
				unsafe_move(key,unified);		
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
			return one; 
};

template<typename Object,typename Guard>
void cordinator<Object,Guard>::ARC::unsafe_move(
			cordinator<Object,Guard>::Key key,
			Memory M)
{
};

template<typename Object,typename Guard>
void cordinator<Object,Guard>::ARC::arc_body(
			cordinator<Object,Guard>::Key			key,
			cordinator<Object,Guard>::ARC::cases	Case)
{
};

template<typename Object,typename Guard>
void cordinator<Object,Guard>::ARC::remove(
			cordinator<Object,Guard>::Key			key,
			cordinator<Object,Guard>::ARC::cases	Case	)
{
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
};

template<typename Object,typename Guard>
void cordinator<Object,Guard>::ARC::replace()
{
};

template<typename Object,typename Guard>
void cordinator<Object,Guard>::ARC::pinned_request()
{
};
//------------------------------------------------
template<typename Object,typename Guard>
void cordinator<Object,Guard>::ARC::push_front_t1(
			cordinator<Object,Guard>::Key key)
{
};
template<typename Object,typename Guard>
void cordinator<Object,Guard>::ARC::push_front_t2(
			cordinator<Object,Guard>::Key key)
{
};
template<typename Object,typename Guard>
void cordinator<Object,Guard>::ARC::push_front_b1(
			cordinator<Object,Guard>::Key key)
{
};
template<typename Object,typename Guard>
void cordinator<Object,Guard>::ARC::push_front_b2(
			cordinator<Object,Guard>::Key key)
{
};
//-------------------------------------------------
template<typename Object,typename Guard>
void cordinator<Object,Guard>::ARC::pop_back_t1(
			cordinator<Object,Guard>::Key key)
{
};
template<typename Object,typename Guard>
void cordinator<Object,Guard>::ARC::pop_back_t2(
			cordinator<Object,Guard>::Key key)
{
};
template<typename Object,typename Guard>
void cordinator<Object,Guard>::ARC::pop_back_b1(
			cordinator<Object,Guard>::Key key)
{
};
template<typename Object,typename Guard>
void cordinator<Object,Guard>::ARC::pop_back_b2(
			cordinator<Object,Guard>::Key key)
{
};
//-------------------------------------------------
template<typename Object,typename Guard>
void cordinator<Object,Guard>::ARC::pop_back_pinned(
			cordinator<Object,Guard>::Key key)
{
};
template<typename Object,typename Guard>
void cordinator<Object,Guard>::ARC::push_front_pinned(
			cordinator<Object,Guard>::Key key)
{
};

//----------------------------------------------------
template<typename Object,typename Guard>
size_t cordinator<Object,Guard>::ARC::LRU_byte_size(		
			cordinator<Object,Guard>::ARC::LRU_list& list)
{
	return 1;
};

#endif			
