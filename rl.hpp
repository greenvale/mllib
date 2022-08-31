/*
Markov chain classes for reinforcement learning
William Denny, 30th August 2020
*/

#pragma once

#include <vector>
#include <tuple>
#include <assert.h>
#include <string>

#include <probability.hpp>

namespace rl 
{

    /* 
    ============================================================================================
        ACTION
        - Action object for a state in an MDP
        - Action is a stochastic process governed by transition distribution
        - Reward is assigned to each state
    */ 

    class Action
    {
    private:
        std::vector<double> m_transDist;
        std::vector<double> m_rewards;
        
    public:
        Action();
        Action(const std::vector<double>& transDist, const std::vector<double>& rewards);
        
        std::tuple<int, double> take();
    };

    Action::Action()
    {

    }

    Action::Action(const std::vector<double>& transDist, const std::vector<double>& rewards)
    {
        assert(transDist.size() == rewards.size());
        m_transDist = transDist;
        m_rewards = rewards;
    }

    std::tuple<int, double> Action::take()
    {
        int stateIndex = mathlib::Probability::discreteEvent(m_transDist);
        double reward = m_rewards[stateIndex];
        return std::make_tuple(stateIndex, reward);
    }

    /* 
    ============================================================================================
        ENVIRONMENT
        - Uses Markov Decision Process architecture
        - Each state has a list of actions, each of which move to a new state through a stochastic process
        - Taking an action index given state index returns a reward and new state index
    */ 

    template <class T>
    class Environment
    {

    private:
        std::vector<T> m_states;
        std::vector<std::vector<Action>> m_actions;

    public:
        Environment();
        Environment(std::vector<T> states, std::vector<std::vector<Action>> actions);
        
        int getNumStates();
        int getNumActions(const int& stateIndex);
        std::tuple<int, double> takeAction(const int& stateIndex, const int& actionIndex);
    };

    template <class T>
    Environment<T>::Environment() 
    {

    }

    template <class T>
    Environment<T>::Environment(std::vector<T> states, std::vector<std::vector<Action>> actions)
    {
        m_states = states;
        m_actions = actions;
    }

    template <class T>
    int Environment<T>::getNumStates()
    {
        return m_states.size();
    }

    template <class T>
    int Environment<T>::getNumActions(const int& stateIndex)
    {
        return m_actions[stateIndex].size();
    }

    template <class T>
    std::tuple<int, double> Environment<T>::takeAction(const int& stateIndex, const int& actionIndex)
    {
        return m_actions[stateIndex][actionIndex].take();
    }

    /* 
    ============================================================================================
        POLICY
        - Stochastic policy governed by vector of distributions for each state
    */

    class Policy
    {
    private:
        std::vector<std::vector<double>> m_policyDist;
    public:
        Policy();
        Policy(const std::vector<std::vector<double>>& policyDist);
        
        int execute(const int& stateIndex);
        void iterateEpsilonGreedy(const std::vector<std::vector<double>>& actionValues, const double& epsilon);
        
        void printPolicyDist();
    };

    Policy::Policy()
    {

    }

    Policy::Policy(const std::vector<std::vector<double>>& policyDist)
    {
        m_policyDist = policyDist;
    }

    int Policy::execute(const int& stateIndex)
    {
        if (m_policyDist[stateIndex].size() > 0)
        {
            return mathlib::Probability::discreteEvent(m_policyDist[stateIndex]);
        }
        else
        {
            return -1; // if no actions available for this state
        }
    }

    void Policy::iterateEpsilonGreedy(const std::vector<std::vector<double>>& actionValues, const double& epsilon)
    {
        assert(actionValues.size() == m_policyDist.size()); // ensure same dimensions between actionValues and policyDist
        
        // loop through states
        for (int i = 0; i < m_policyDist.size(); ++i)
        {
            assert(actionValues[i].size() == m_policyDist[i].size()); // ensure same dimensions between actionValues and policyDist
            
            int numActions = m_policyDist[i].size();
            
            // obtain action index with maximum action value
            int maxActionValueIndex = 0;
            double maxActionValue;
            for (int j = 0; j < numActions; ++j)
            {
                if (actionValues[i][j] > actionValues[i][maxActionValueIndex])
                {
                    maxActionValueIndex = j;
                    maxActionValue = actionValues[i][j];
                }
            }
            
            // change policy distribution for this state
            for (int j = 0; j < numActions; ++j)
            {
                if (j == maxActionValueIndex)
                {
                    m_policyDist[i][j] = 1 - epsilon + (epsilon/numActions);
                }
                else
                {
                    m_policyDist[i][j] = epsilon/numActions;
                }
            }
        }
    }

    /*
    Print policy distribution
    */
    void Policy::printPolicyDist()
    {
        std::cout << "Policy distribution:" << std::endl;
        
        for (int i = 0; i < m_policyDist.size(); ++i)
        {
            std::cout << "State index " << i << ": { ";
            
            for (int j = 0; j < m_policyDist[i].size(); ++j)
            {
                if (j >= 1)
                {
                    std::cout << ", ";
                }
                std::cout << m_policyDist[i][j];
            }
            
            std::cout << " }" << std::endl;    
        }
    }

    /* 
    ============================================================================================
        AGENT
    */

    template <class T>
    class Agent
    {
    private:
        Policy m_policy;
        std::vector<double> m_stateValues;
        std::vector<std::vector<double>> m_actionValues;
        Environment<T>* m_environmentPtr;
    public:
        Agent();
        Agent(Environment<T>* environmentPtr, const Policy& policy);
        
        double sample(const int& startStateIndex);
        
        void evalStateValue_MC(const double& discountFactor, const double& alpha, const double& tol, const int& maxEpisodes);
        void evalStateValue_TD(const double& discountFactor, const double& alpha, const double& tol, const int& maxEpisodes);
        
        void optimisePolicy_SARSA(const double& discountFactor, const double& alpha, const double& epsilon, const int& maxEpisodes);
        
        void printStateValues();
        void printActionValues();
    };

    /* 
    ================================
    */

    template <class T>
    Agent<T>::Agent() 
    {

    }

    template <class T>
    Agent<T>::Agent(Environment<T>* environmentPtr, const Policy& policy)
    {
        m_environmentPtr = environmentPtr;
        m_policy = policy;
    }

    /*
    Sample the MDP environment using agent, returns the accumulated reward
    */
    template <class T>
    double Agent<T>::sample(const int& startStateIndex)
    {
        int stateIndex = startStateIndex;
        double totalReward = 0.0;
        
        int flag = 0;
        
        while (flag == 0)
        {
            // execute policy for current state
            int actionIndex = m_policy.execute(stateIndex);
            
            if (actionIndex == -1)
            {
                return totalReward; // terminate sample if reached end state (with no actions)
            }
            
            // take action to get new state and reward
            std::tuple<int, double> outcome = m_environmentPtr->takeAction(stateIndex, actionIndex);
            
            int newStateIndex = std::get<0>(outcome);
            double reward = std::get<1>(outcome);
            
            // update state and return
            stateIndex = newStateIndex;
            totalReward += reward;
        }
        return 0.0;
    }

    /*
    Evaluates the state value function using the Monte Carlo method
    This implementation does not use eligibility traces!
    */
    template <class T>
    void Agent<T>::evalStateValue_MC(const double& discountFactor, const double& alpha, const double& tol, const int& maxEpisodes)
    {
        m_stateValues = std::vector<double>(m_environmentPtr->getNumStates(), 0.0);
        
        // loop through episodes
        int evalFlag = 0;
        int episodeNum = 0;
        while (evalFlag == 0)
        {
            std::vector<bool> visited(m_environmentPtr->getNumStates(), 0);
            std::vector<double> stateReturns(m_environmentPtr->getNumStates(), 0.0);
            
            int stateIndex = 0;
            
            // loop through states
            int episodeFlag = 0;
            while (episodeFlag == 0)
            {
                // note visit of current state
                visited[stateIndex] = 1;
                
                int actionIndex = m_policy.execute(stateIndex);
                
                if (actionIndex == -1)
                {
                    episodeFlag = 1; // finish episode if no actions available
                }
                
                if (episodeFlag == 0)
                {
                    // take action to get new state and reward
                    std::tuple<int, double> outcome = m_environmentPtr->takeAction(stateIndex, actionIndex);
                    
                    int newStateIndex = std::get<0>(outcome);
                    double reward = std::get<1>(outcome);
                    
                    // add reward to all states visited as this counts as their return
                    for (int i = 0; i < m_environmentPtr->getNumStates(); ++i)
                    {
                        if (visited[i] == 1)
                        {
                            stateReturns[i] *= discountFactor; // discount existing return
                            stateReturns[i] += reward;  
                        }
                    }
                    
                    // update state
                    stateIndex = newStateIndex;
                }
            }
            
            double diffTotal = 0.0;
            
            // following sample, adjust the value function
            for (int i = 0; i < m_environmentPtr->getNumStates(); ++i)
            {
                diffTotal += (stateReturns[i] - m_stateValues[i]) * (stateReturns[i] - m_stateValues[i]);
                m_stateValues[i] += alpha * (stateReturns[i] - m_stateValues[i]);
            }
            
            if ((diffTotal < tol) || (episodeNum + 1 >= maxEpisodes))
            {
                std::cout << "Evaluated state value function using Monte Carlo in " << episodeNum + 1 << " steps with final diff " << diffTotal << std::endl;
                evalFlag = 1;
            }
            
            episodeNum++;
        }
    }

    /*
    Evaluate state value function using temporal difference
    */
    template <class T>
    void Agent<T>::evalStateValue_TD(const double& discountFactor, const double& alpha, const double& tol, const int& maxEpisodes)
    {
        // initialise state value function
        m_stateValues = std::vector<double>(m_environmentPtr->getNumStates(), 0.0);
        std::vector<double> stateValuesPrev = m_stateValues; // create prev copy of state values - to measure convergence
        
        // loop through episodes
        int evalFlag = 0;
        int episodeNum = 0;
        while (evalFlag == 0)
        {
            int stateIndex = 0;
            
            // loop through states
            int episodeFlag = 0;
            while (episodeFlag == 0)
            {
                int actionIndex = m_policy.execute(stateIndex);
                
                if (actionIndex == -1)
                {
                    episodeFlag = 1; // finish sample loop if no actions available
                }
                
                if (episodeFlag == 0)
                {
                    // take action to get new state and reward
                    std::tuple<int, double> outcome = m_environmentPtr->takeAction(stateIndex, actionIndex);
                    
                    int newStateIndex = std::get<0>(outcome);
                    double reward = std::get<1>(outcome);
                    
                    // add reward to appropriate state
                    m_stateValues[stateIndex] += alpha * ((reward + discountFactor * m_stateValues[newStateIndex]) - m_stateValues[stateIndex]);
                    
                    // update state
                    stateIndex = newStateIndex;
                }
            }
            
            double diffTotal = 0.0;
            
            // following sample, calculate change in state value function to measure convergence
            for (int i = 0; i < m_environmentPtr->getNumStates(); ++i)
            {
                diffTotal += (m_stateValues[i] - stateValuesPrev[i]) * (m_stateValues[i] - stateValuesPrev[i]);
            }
            
            stateValuesPrev = m_stateValues; // update copy of state value function
            
            if ((diffTotal < tol) || (episodeNum + 1 >= maxEpisodes))
            {
                std::cout << "Evaluated state value function using Temporal Difference in " << episodeNum + 1 << " steps with final diff " << diffTotal << std::endl;
                evalFlag = 1;
            }
            
            episodeNum++;
        }
    }

    /*
    Optimise policy
        - Calculates action value function
        - Optimises policy with SARSA algorithm
    */
    template <class T>
    void Agent<T>::optimisePolicy_SARSA(const double& discountFactor, const double& alpha, const double& epsilon, const int& maxEpisodes) 
    {
        // initialise action value function
        m_actionValues = {};
        for (int i = 0; i < m_environmentPtr->getNumStates(); ++i)
        {
            std::vector<double> row(m_environmentPtr->getNumActions(i), 0.0);
            m_actionValues.push_back(row);
        }
        
        // loop through episodes
        int iterFlag = 0;
        int episodeNum = 0;
        
        while (iterFlag == 0)
        {
            
            int stateIndex = 0;
            int actionIndex = m_policy.execute(stateIndex);
            
            int episodeFlag = 0;
            while (episodeFlag == 0)
            {
                // take action
                std::tuple<int, double> outcome = m_environmentPtr->takeAction(stateIndex, actionIndex);
                
                int newStateIndex = std::get<0>(outcome);
                double reward = std::get<1>(outcome);
                
                // update policy using greedy method with existing Q values
                m_policy.iterateEpsilonGreedy(m_actionValues, epsilon);
                
                // choose next action
                int newActionIndex = m_policy.execute(newStateIndex);
                
                if (newActionIndex == -1)
                {
                    episodeFlag = 1; // there are no actions to take - end episode as now in end state
                }
                
                if (episodeFlag == 0)
                {
                    // calculate new Q value for current state
                    m_actionValues[stateIndex][actionIndex] += alpha * (reward + (discountFactor * m_actionValues[newStateIndex][newActionIndex]) - m_actionValues[stateIndex][actionIndex]); 
                    
                    // step system
                    stateIndex = newStateIndex;
                    actionIndex = newActionIndex;
                }
            }
            
            if (episodeNum + 1 >= maxEpisodes)
            {
                iterFlag = 1;
            }
            
            episodeNum++;
        }
        
        // print final policy
        m_policy.printPolicyDist();
    }

    /*
    Print state value function
    */
    template <class T>
    void Agent<T>::printStateValues()
    {   
        std::cout << "State value function:" << std::endl;
        
        for (int i = 0; i < m_environmentPtr->getNumStates(); ++i)
        {
            std::cout << "State index: " << m_stateValues[i] << std::endl;
        }
    }

    /*
    Print action value function
    */
    template <class T>
    void Agent<T>::printActionValues()
    {
        std::cout << "Action value function:" << std::endl;
        
        for (int i = 0; i < m_environmentPtr->getNumStates(); ++i)
        {
            std::cout << "State index " << i << ": { ";
            
            for (int j = 0; j < m_environmentPtr->getNumActions(i); ++j)
            {
                if (j >= 1)
                {
                    std::cout << ", ";
                }
                std::cout << m_actionValues[i][j];
            }
            
            std::cout << " }" << std::endl;    
        }
    }
}
