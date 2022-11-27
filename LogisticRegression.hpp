#pragma once

#include "../mathlib/LinearAlgebra.hpp"
#include "../mathlib/probability.hpp"
#include "assert.h"
#include <vector>
#include <iostream>
#include <functional>
#include "math.h"

class LogisticReg
{

public:

    unsigned int m_numInputs;
    unsigned int m_numOutputs;
    mathlib::Matrix m_weight;
    mathlib::Matrix m_bias;
    
public:

    LogisticReg() = delete;
    LogisticReg(const unsigned int& numInputs, const unsigned int& numOutputs) : 
        m_numInputs(numInputs), 
        m_numOutputs(numOutputs)
    {
        assert(numInputs > 0);
        assert(numOutputs > 0);
        
        m_weight    = mathlib::Matrix({m_numInputs, m_numOutputs});
        m_bias      = mathlib::Matrix({m_numOutputs, 1});
        
        auto randomise = []() { return  mathlib::Probability::randomRealNumber(); }; // lambda expression for randomisation
        m_weight.operation(randomise);
        m_bias.operation(randomise);
    }

    mathlib::Matrix evaluate(const mathlib::Matrix& input)
    {
        assert(input.size()[0] == m_numInputs);
        assert(input.size()[1] == 1);

        mathlib::Matrix preactivation = m_weight.transpose() * input + m_bias;
        
        auto activation = [](double x){ return 1.0 / (1.0 + exp(-1.0 * x)); }; // activation function lambda expression
        mathlib::Matrix output = preactivation;
        output.operation(activation);
        return output;
    }

    void train(const mathlib::Matrix& trainingInput, const mathlib::Matrix& trainingOutput, double learningRate, double tol, unsigned int maxIter)
    {
        assert(trainingInput.size()[0] == m_numInputs);
        assert(trainingInput.size()[1] == 1);
        assert(trainingOutput.size()[0] == m_numOutputs);
        assert(trainingOutput.size()[1] == 1);

        for (int i = 0; i < maxIter; ++i)
        {
            mathlib::Matrix preactivation = m_weight.transpose() * trainingInput + m_bias;
            
            auto activation = [](double x){ return 1.0 / (1.0 + exp(-1.0 * x)); }; // activation function lambda expression
            mathlib::Matrix output = preactivation;
            output.operation(activation);

            auto activationDiff = [](double x){ return x * (1.0 - x); };

            // loss J
            double loss = 0.0;
            for (unsigned int i = 0; i < m_numOutputs; ++i)
            {
                // loss function for regression: -y*log(a) - (1-y)*log(1-a)
                loss += -1.0 * (trainingOutput.get({i,0})*log(output.get({i,0})) + (1.0 - trainingOutput.get({i,0}))*log(1.0 - output.get({i,0})));
            }
            std::cout << "Iteration (" << i << ") - Loss: " << loss << std::endl;

            // dJ/da Jacobian
            mathlib::Matrix dJ_da({1, m_numOutputs});
            for (unsigned int i = 0; i < m_numOutputs; ++i)
            {
                // derivative of loss function for logistic regression: -y/a + (1-y)/(1-a)
                dJ_da.set({0,i}, -(trainingOutput.get({i,0}) / output.get({i,0})) + ((1.0 - trainingOutput.get({i,0})) / (1.0 - output.get({i,0}))));
            }

            mathlib::Matrix chainDeriv = dJ_da; // accumulate chain derivative

            // Calculate da/dz Jacobian
            mathlib::Matrix da_dz({m_numOutputs, m_numOutputs}, 0.0);
            for (unsigned int i = 0; i < m_numOutputs; ++i)
            {
                da_dz.set({i,i}, output.get({i,0}) * (1.0 - output.get({i,0})));
            }
            chainDeriv = chainDeriv * da_dz;

            // dJ/db 
            mathlib::Matrix db = chainDeriv.transpose();

            // dZ/dW is 3D and is therefore stored as array numInput long of (numOutput x numOutput) matrices
            mathlib::Matrix dW({m_numInputs, m_numOutputs});

            for (unsigned int i = 0; i < m_numInputs; ++i)
            {
                // Calculate dz/dW Jacobian
                mathlib::Matrix dz_dW_byInput({m_numOutputs, m_numOutputs}, 0.0);
                for (unsigned int j = 0; j < m_numOutputs; ++j)
                {
                    dz_dW_byInput.set({j, j}, trainingInput.get({i,0}));
                }

                dW.setRegion({i,0}, chainDeriv * dz_dW_byInput);
            }

            m_weight += -learningRate * dW;
            m_bias += -learningRate * db;

            std::cout << "Weights: " << std::endl;
            m_weight.display(); 
            std::cout << "Biases: " << std::endl;
            m_bias.display();
            std::cout << std::endl;

            if (loss < tol)
            {
                std::cout << "Exited training loop after " << i << " iterations" << std::endl;
                break;
            }
        }
    }

};