#include <iostream>
#include <vector> 
#include "mdp.h"


int main()
{
	float discountFactor = 1.0;
	std::vector<float> stateSpace = {0.0, 1.0, 2.0, 3.0, 4.0};
	std::vector<std::vector<std::vector<float>>> actionSpace = {
		// actions for state 0.0
		{
			{0.0, 1.0, 0.0, 0.0, 0.0} 
		},
		// actions for state 1.0
		{
			{0.0, 0.0, 1.0, 0.0, 0.0} 
		},
		// actions for state 2.0
		{
			{0.0, 0.5, 0.0, 0.5, 0.0} 
		},
		// actions for state 3.0
		{
			{0.0, 0.0, 0.0, 0.0, 1.0} 
		},
		// actions for state 4.0
		{
		}
	};
	std::vector<std::vector<float>> rewardSpace = {
		{ 1.0 },
		{ 1.0 },
		{ 1.0 },
		{ 1.0 },
		{ 1.0 }
	};
	
	Mdp myMdp = Mdp(discountFactor, stateSpace, actionSpace, rewardSpace);

	myMdp.runEpisode();
	
}