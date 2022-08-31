#include "environment.h"
#include "agent.h"
#include "mathlib.h"
#include <iostream>
#include <tuple>
#include <vector>
#include <stdexcept>
#include <fstream>

int main()
{
	std::vector<float> stateSpace = { 0.0, 0.25, 0.5, 0.75, 1.0 };
	std::vector<std::vector<std::vector<float>>> actionSpace = {
		{
			{0.0, 1.0, 0.0, 0.0, 0.0},
		},
		{
			{1.0, 0.0, 0.0, 0.0, 0.0},
			{0.0, 0.0, 1.0, 0.0, 0.0}
		},
		{
			{0.0, 1.0, 0.0, 0.0, 0.0},
			{0.0, 0.0, 0.0, 1.0, 0.0}
		},
		{
			{0.0, 0.0, 1.0, 0.0, 0.0},
			{0.0, 0.0, 0.0, 0.0, 1.0}
		},
		{
		}
	};
	std::vector<std::vector<std::vector<float>>> rewardSpace = {
		{
			{0.0, -1.0, 0.0, 0.0, 0.0},
		},
		{
			{-1.0, 0.0, 0.0, 0.0, 0.0},
			{0.0, 0.0, -1.0, 0.0, 0.0},
		},
		{
			{0.0, -1.0, 0.0, 0.0, 0.0},
			{0.0, 0.0, 0.0, -1.0, 0.0},
		},
		{
			{0.0, 0.0, -1.0, 0.0, 0.0},
			{0.0, 0.0, 0.0, 0.0, -1.0},
		},
		{
		},
	};

	Environment myEnvironment = Environment(stateSpace, actionSpace, rewardSpace);
	unsigned int initialStateIndex = 0;
	float initialState = stateSpace[initialStateIndex];
	float alpha = 0.01;
	float discountFactor = 1.0;

	Agent myAgent = Agent(stateSpace.size(), myEnvironment, alpha, discountFactor);

	std::vector<float> stateValue(stateSpace.size(), 0.0);

	std::ofstream myFile;
	myFile.open("stateValue.csv", std::ios_base::trunc);

	unsigned int numEpisodes = 10000;

	for (int i = 0; i < numEpisodes; ++i)
	{
		myAgent.runEpisode(initialStateIndex, initialState);

		// print state value function
		stateValue = myAgent.getStateValue();

		for (int i = 0; i < stateSpace.size() - 1; ++i)
		{
			myFile << stateValue[i] << ",";
		}
		myFile << stateValue[stateSpace.size() - 1];
		
		if (i < numEpisodes - 1)
			myFile << std::endl;
	}
}