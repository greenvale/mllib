#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <vector>
#include <tuple>

class Environment
{
private:

	std::vector<float> stateSpace;
	std::vector<std::vector<std::vector<float>>> actionSpace;
	std::vector<std::vector<std::vector<float>>> rewardSpace;

public:

	Environment();
	Environment(
		const std::vector<float>&, 
		const std::vector<std::vector<std::vector<float>>>&, 
		const std::vector<std::vector<std::vector<float>>>&
	);
	Environment(const Environment&);
	~Environment();

	std::tuple<unsigned int, float, unsigned int, float> takeAction(const unsigned int&, const unsigned int&);

};

#endif