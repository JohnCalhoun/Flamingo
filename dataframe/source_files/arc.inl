//arc.inl
#include "cordinator.cpp"

template<typename Object,typename Guard>
void cordinator<Object,Guard>::ARC::request(
	cordinator<Object,Guard>::Key key,
	Memory M)
{
	//assumptions
	//	write access to the key dataframe

	lock_guard guard(*mutex); 
	Value dataframe_ptr=get_ptr(Key); 
	if(dataframe_ptr->location() != M){
		cases Case=find_case(key); 
		remove(key,Case); 

		switch M:
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
			if(data_frame_ptr->location() = device){
				unsafe_move(key,host);
			}
		}
		case unified:
		{
			unsafe_move(key,unified);		
		}
	}
}			
