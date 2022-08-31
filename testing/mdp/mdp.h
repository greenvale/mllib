#include <vector>

class Mdp
{

private:

	float currentState;
	unsigned int currentStateIndex;
	float futureState;
	unsigned int futureStateIndex;
	unsigned int actionIndex;
	float reward;
	
	std::vector<float> episodeStateHistory;
	std::vector<unsigned int> episodeActionHistory;
	std::vector<float> episodeRewardHistory;
	float episodeReturn;

	float discountFactor;
	std::vector<float> stateSpace;
	std::vector<std::vector<std::vector<float>>> actionSpace;
	std::vector<std::vector<float>> rewardSpace;

public:

	//ctor
	Mdp();
	Mdp(float, std::vector<float>, std::vector<std::vector<std::vector<float>>>, std::vector<std::vector<float>>);
	//dtor
	~Mdp();
	
	unsigned int randomEvent(std::vector<float>);
	float randomNumber();
	void runEpisode();
	
	std::vector<float> getEpisodeStateHistory();
	std::vector<unsigned int> getEpisodeActionHistory();
	std::vector<float> getEpisodeRewardHistory();
	float getEpisodeReturn();

};