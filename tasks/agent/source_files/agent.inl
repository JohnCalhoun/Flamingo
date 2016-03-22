//agent.inl
#include <agent.cpp>

template<class DataFrame>	
agent<DataFrame>::agent(	DataFrame& d,
					agent<DataFrame>::Func& f,
					agent<DataFrame>::Init& i){
	init(data); 
};
template<class DataFrame>	
void agent<DataFrame>::operator()(){

}; 
