#ifndef MATHLIB_H
#define MATHLIB_H

#include <vector>

namespace MathLib
{
	class Probability
	{
	public:
		// returns random number between 0 and 1
		static float randomNumber();

		// runs random event with discrete outcomes given probabilities of each outcome
		static unsigned int randomDiscreteEvent(const std::vector<float>&);
	};
}

#endif