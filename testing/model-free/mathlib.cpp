#include "mathlib.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <stdexcept>

namespace MathLib
{
	float Probability::randomNumber()
	{
		return (float)rand() / RAND_MAX;
	}

	unsigned int Probability::randomDiscreteEvent(const std::vector<float>& prob)
	{
		// create vector of event boundaries and cumulative variable
		std::vector<float> eventBoundaries(prob.size() + 1, 0.0);
		float cumulat = 0.0;

		// calculate event region boundaries
		for (unsigned int i = 0; i < prob.size(); ++i)
		{
			cumulat += prob[i];
			eventBoundaries[i + 1] = cumulat;
		}

		// generate random number
		bool eventDone = false;
		float randomNum;

		while (eventDone == false)
		{
			randomNum = randomNumber();

			if ((randomNum > 0.0) && (randomNum < 1.0))
			{
				eventDone = true;
			}
		}

		// identify the region index that the random number lies in
		for (unsigned int i = 0; i < prob.size(); ++i)
		{
			if ((randomNum >= eventBoundaries[i]) && (randomNum < eventBoundaries[i + 1]))
			{
				return i;
			}
		}

		throw std::invalid_argument("Error in random discrete event");
	}
}