#include <iostream>
#include <tuple>

#include "../rl.hpp"

struct EnvironmentTemplate
{ 
    std::vector<double> states;
    std::vector<std::vector<rl::Action>> actions;
};

int main()
{

    EnvironmentTemplate myGridWorld;
    myGridWorld.states = { 0.0, 1.0, 2.1, 2.2, 3.0, 4.0 };
    
    myGridWorld.actions = {
        // state 0.0
        { rl::Action({0.0, 1.0, 0.0, 0.0, 0.0, 0.0}, {0.0, -1.0, 0.0, 0.0, 0.0, 0.0}) },
        
        // state 1.0
        { rl::Action({0.0, 0.0, 0.2, 0.8, 0.0, 0.0}, {0.0, -1.0, 10.0, -5.0, 0.0, 0.0}),
          rl::Action({0.0, 0.0, 0.4, 0.6, 0.0, 0.0}, {0.0, -1.0, 5.0, -5.0, 0.0, 0.0}) },
          
        // state 2.1 
        { rl::Action({0.0, 0.0, 0.0, 0.0, 1.0, 0.0}, {0.0, 0.0, 0.0, 0.0, -1.0, 0.0}) },
        
        // state 2.2        
        { rl::Action({0.0, 0.0, 0.0, 0.0, 1.0, 0.0}, {0.0, 0.0, 0.0, 0.0, -1.0, 0.0}) },
        
        // state 3.0
        { rl::Action({0.0, 0.0, 0.0, 0.0, 0.0, 1.0}, {0.0, 0.0, 0.0, 0.0, 0.0, -1.0}) },
        
        {}
    };
    
    rl::Environment<double> myEnvironment = rl::Environment<double>(myGridWorld.states, myGridWorld.actions);
    
    rl::Policy myPolicy = rl::Policy({{1.0}, {0.5, 0.5}, {1.0}, {1.0}, {1.0}, {}});
    
    rl::Agent<double> myAgent = rl::Agent<double>(&myEnvironment, myPolicy);
    
    myAgent.evalStateValue_TD(0.1, 0.1, 0.001, 100000);
    
    myAgent.printStateValues();
    
    myAgent.optimisePolicy_SARSA(0.1, 0.1, 0.1, 100000);
    
    myAgent.printActionValues();
    
}
