#include "mathlib.h"
#include "environment.h"
#include "agent.h"
#include <iostream>
#include <tuple>
#include <vector>

// constructors
// empty ctor
Agent::Agent()
{

}

// initialisation ctor
Agent::Agent(const unsigned int& newNumStates, const Environment& newEnvironment, const float& newAlpha, const float& newDiscountFactor)
{
	numStates = newNumStates;
	environment = newEnvironment;
	alpha = newAlpha;
	discountFactor = newDiscountFactor;

	stateValue = std::vector<float>(numStates, 0.0);
}

// copy ctor
Agent::Agent(const Agent& newAgent)
{

}

// destructor
Agent::~Agent()
{

}

// ============================================

bool Agent::updateAgent(const std::tuple<unsigned int, float, unsigned int, float>& environmentResults)
{
	// obtain new state from environment results
	unsigned int futureStateIndex;
	float futureState;
	unsigned int futureActionSpaceSize;
	float reward;

	futureStateIndex = std::get<0>(environmentResults);
	futureState = std::get<1>(environmentResults);
	futureActionSpaceSize = std::get<2>(environmentResults);
	reward = std::get<3>(environmentResults);

	// update state value function
	stateValue[currentStateIndex] += alpha * ((stateValue[futureStateIndex] * discountFactor + reward) - stateValue[currentStateIndex]);

	// add new values to history
	episodeStateIndexHistory.push_back(futureStateIndex);
	episodeRewardHistory.push_back(reward);
	
	// update current state and action space size
	currentStateIndex = futureStateIndex;
	currentState = futureState;
	currentActionSpaceSize = futureActionSpaceSize;

	// if there are no actions available, the episode will terminate
	return (currentActionSpaceSize <= 0);
}

unsigned int Agent::decideAction()
{
	unsigned int actionIndex;

	// Makeshift policy:
	std::vector<float> statePolicyProb(currentActionSpaceSize, 1.0 / currentActionSpaceSize);

	// Make decision by running event with policy probabilities
	actionIndex = MathLib::Probability::randomDiscreteEvent(statePolicyProb);

	//std::cout << "Action index: " << actionIndex << std::endl;

	return actionIndex;
}

void Agent::runEpisode(const unsigned int& initialStateIndex, const float& initialState)
{
	// set initial state
	currentStateIndex = initialStateIndex;
	currentState = initialState;
	currentActionSpaceSize = 0;

	// clear episode history
	episodeStateIndexHistory.clear();
	episodeRewardHistory.clear();

	// set running variables
	unsigned int actionIndex;
	bool terminated = false;
	int n = 0;
	std::tuple<unsigned int, float, unsigned int, float> environmentResults = { currentStateIndex, currentState, currentActionSpaceSize, 0.0 };

	while (terminated == false)
	{
		// decide the next action
		if (n > 0)
		{
			actionIndex = decideAction();
		}
		else
		{
			actionIndex = 0;
		}

		//std::cout << "Action for step " << n + 1 << std::endl;

		// take action in the environment to get action space
		environmentResults = environment.takeAction(currentStateIndex, actionIndex);

		// update agent
		terminated = updateAgent(environmentResults);

		//std::cout << "New state: " << currentState << std::endl;

		n++;
	}
}

// ============================================
// GETTER FUNCTIONS

std::vector<float> Agent::getStateValue()
{
	return stateValue;
}