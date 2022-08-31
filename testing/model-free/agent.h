#ifndef AGENT_H
#define AGENT_H

#include "environment.h"
#include <vector>

class Agent
{
private:
	
	float alpha;
	float discountFactor;
	
	unsigned int numStates;
	std::vector<float> stateValue;

	Environment environment;

	unsigned int currentStateIndex;
	float currentState;
	unsigned int currentActionSpaceSize;

	std::vector<unsigned int> episodeStateIndexHistory;
	std::vector<float> episodeRewardHistory;
	std::vector<std::vector<float>> episodeStateValueHistory;
	
public:

	Agent();
	Agent(const unsigned int&, const Environment&, const float&, const float&);
	Agent(const Agent&);
	~Agent();

	bool updateAgent(const std::tuple<unsigned int, float, unsigned int, float>&);
	unsigned int decideAction();
	void runEpisode(const unsigned int&, const float&);

	std::vector<float> getStateValue();
};

#endif