#include "environment.h"
#include "mathlib.h"
#include <iostream>
#include <vector>
#include <tuple>
#include <stdexcept>


// constructors
// empty ctor
Environment::Environment()
{

}

// initialisation ctor
Environment::Environment(
	const std::vector<float>& newStateSpace,
	const std::vector<std::vector<std::vector<float>>>& newActionSpace,
	const std::vector<std::vector<std::vector<float>>>& newRewardSpace
)
{
	stateSpace = newStateSpace;
	actionSpace = newActionSpace;
	rewardSpace = newRewardSpace;
}

// copy ctor
Environment::Environment(const Environment& environment)
{
	stateSpace = environment.stateSpace;
	actionSpace = environment.actionSpace;
	rewardSpace = environment.rewardSpace;
}

// destructor
Environment::~Environment()
{

}

// ============================================

// action function
std::tuple<unsigned int, float, unsigned int, float> Environment::takeAction(const unsigned int& currentStateIndex, const unsigned int& actionIndex)
{
	unsigned int futureStateIndex;
	float futureState;
	unsigned int futureActionSpaceSize;
	float reward;

	// get future state probabilities given action
	std::vector<float> futureStateProb = actionSpace[currentStateIndex][actionIndex];

	// run random event to get future state given action
	futureStateIndex = MathLib::Probability::randomDiscreteEvent(futureStateProb);
	futureState = stateSpace[futureStateIndex];

	// get reward given current state, action and future state
	reward = rewardSpace[currentStateIndex][actionIndex][futureStateIndex];

	// get number of actions avaiable in future state
	futureActionSpaceSize = actionSpace[futureStateIndex].size();

	return std::make_tuple(futureStateIndex, futureState, futureActionSpaceSize, reward);
}