#include "mdp.h"
#include <iostream>
#include <vector>

//ctors
Mdp::Mdp()
{
	std::cout << "Empty constructor invoked" << std::endl;
}

Mdp::Mdp(float _discountFactor, std::vector<float> _stateSpace, std::vector<std::vector<std::vector<float>>> _actionSpace, std::vector<std::vector<float>> _rewardSpace)
{
	discountFactor = _discountFactor;
	stateSpace = _stateSpace;
	actionSpace = _actionSpace;
	rewardSpace = _rewardSpace;
}
//dtor
Mdp::~Mdp()
{
}

//===============================================================

// generate random number between 0 and 1
float Mdp::randomNumber()
{
	return (float)rand() / RAND_MAX;
}

// return index of region given probability distribution of random event
unsigned int Mdp::randomEvent(std::vector<float> distrib)
{
	// create vector of event boundaries and cumulative variable
	std::vector<float> eventBoundaries(distrib.size() + 1, 0.0);
	float cumulat = 0.0;
	
	// calculate event region boundaries
	for (unsigned int i = 0; i < distrib.size(); ++i)
	{
		cumulat += distrib[i];
		eventBoundaries[i + 1] = cumulat;
	}

	// generate random number
	float randomNum = randomNumber();

	// identify the region index that the random number lies in
	for (unsigned int i = 0; i < distrib.size(); ++i)
	{
		if ((randomNum >= eventBoundaries[i]) && (randomNum < eventBoundaries[i + 1]))
		{
			return i;
		}
	}
}

void Mdp::runEpisode()
{
	// initial state
	currentStateIndex = 0;
	currentState = stateSpace[currentStateIndex];

	bool terminated = false;
	int step = 0;

	while (terminated == false)
	{
		std::cout << "================ STEP: " << step << std::endl;
		std::cout << "Current state index: " << currentStateIndex << std::endl;
		std::cout << "Current state: " << currentState << std::endl;
		
		//get number of actions
		unsigned int num_actions = actionSpace[currentStateIndex].size();

		std::cout << "Number of actions available: " << num_actions << std::endl;

		// if no actions are available, terminate the episode
		if (num_actions == 0)
		{
			terminated = true;
			std::cout << "Episode terminated" << std::endl;
		}

		if (terminated == false)
		{
			// generate policy to be equal chance - TEMPORARY
			std::vector<float> currentPolicyDistrib(num_actions, 1.0 / num_actions);

			// obtain action by running random event with policy probability distribution
			actionIndex = randomEvent(currentPolicyDistrib);

			// get future state probability distribution for action chosen
			std::vector<float> futureStateDistrib = actionSpace[currentStateIndex][actionIndex];

			// get future state by running random event with future state probability distribution
			futureStateIndex = randomEvent(futureStateDistrib);
			futureState = stateSpace[futureStateIndex];

			std::cout << "Future state index: " << futureStateIndex << std::endl;
			std::cout << "Future state: " << futureState << std::endl;

			currentStateIndex = futureStateIndex;
			currentState = futureState;
		}
		
		step++;
	}
}

//===============================================================

std::vector<float> Mdp::getEpisodeStateHistory()
{
	return episodeStateHistory;
}

std::vector<unsigned int> Mdp::getEpisodeActionHistory()
{
	return episodeActionHistory;
}

std::vector<float> Mdp::getEpisodeRewardHistory()
{
	return episodeRewardHistory;
}

float Mdp::getEpisodeReturn()
{
	return episodeReturn;
}
