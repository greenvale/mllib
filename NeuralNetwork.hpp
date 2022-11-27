#pragma once

#include <vector>
#include "assert.h"
#include "../mathlib/LinearAlgebra.hpp"
#include "../mathlib/probability.hpp"

class NeuralNetwork
{
public:
    unsigned int m_numLayers;
    std::vector<unsigned int> m_shape;
    std::vector<mathlib::Matrix*> m_weights;
    std::vector<mathlib::Matrix*> m_biases;
    std::vector<mathlib::Matrix*> m_prelayers;
    std::vector<mathlib::Matrix*> m_layers;

public:
    // lambda expressions for uniform matrix operations
    static constexpr auto randomise = []() { return  mathlib::Probability::randomRealNumber(); }; // lambda expression for randomisation
    static constexpr auto sigmoidActivation = [](double x){ return 1.0 / (1.0 + exp(-1.0 * x)); };
    static constexpr auto sigmoidActivationDiff = [](double x){ return x * (1.0 - x); };

public:
    NeuralNetwork() = delete;
    NeuralNetwork(const std::vector<unsigned int> shape);

    mathlib::Matrix evaluate(const mathlib::Matrix& input);

    double regressLoss(const double& y, const double& a);
    double regressLossDiff(const double& y, const double& a);
    double logisticLoss(const double& y, const double& a);
    double logisticLossDiff(const double& y, const double& a);

    void train(
        const std::vector<std::vector<double>>& trainingInput,
        const std::vector<std::vector<double>>& trainingOutput,
        const double& learningRate,
        const double& tol,
        const unsigned int& maxIter
    );

    void display();
};

/* ctor */
NeuralNetwork::NeuralNetwork(const std::vector<unsigned int> shape) 
{
    assert(shape.size() >= 2); // there must be at least two layers for input and output layers
    this->m_numLayers = shape.size();
    this->m_shape = shape;

    // allocate matrices
    for (int i = 0; i < this->m_numLayers; ++i)
    {
        mathlib::Matrix* layer = new mathlib::Matrix({this->m_shape[i], 1}, 0.0);
        this->m_layers.push_back(layer);

        if (i > 0)
        {
            mathlib::Matrix* weight     = new mathlib::Matrix({this->m_shape[i - 1], this->m_shape[i]});
            mathlib::Matrix* bias       = new mathlib::Matrix({this->m_shape[i], 1});
            mathlib::Matrix* prelayer   = new mathlib::Matrix({this->m_shape[i], 1}, 0.0);
            this->m_weights.push_back(weight);
            this->m_biases.push_back(bias);
            this->m_prelayers.push_back(prelayer);
        }
    }

    // initialise weightings with random numbers
    for (int i = 0; i < this->m_weights.size(); ++i)
    {
        this->m_weights[i]->operation(randomise);
        this->m_biases[i]->operation(randomise);
    }
}

/* evaluation neural network - takes input and returns output */
mathlib::Matrix NeuralNetwork::evaluate(const mathlib::Matrix& input)
{
    *(this->m_layers[0]) = input; // set the input

    for (int i = 1; i < this->m_numLayers; ++i)
    {
        *(this->m_prelayers[i - 1]) = (this->m_weights[i - 1]->transpose() * *(this->m_layers[i - 1])) + *(this->m_biases[i - 1]); 
        *(this->m_layers[i]) = *(this->m_prelayers[i - 1]);
        this->m_layers[i]->operation(sigmoidActivation);
    }

    return *(this->m_layers[this->m_numLayers - 1]); // return output (from output layer)
}

/* regression loss function */
double NeuralNetwork::regressLoss(const double& y, const double& a)
{
    return 0.5 * (y - a) * (y - a);
}

/* differential of regression loss function */
double NeuralNetwork::regressLossDiff(const double& y, const double& a)
{
    return (y - a);
}

/* logistic loss function */
double NeuralNetwork::logisticLoss(const double& y, const double& a)
{
    return -(y*log(a) + (1.0 - y)*log(1.0 - a));
}

/* differential of logistic loss function */
double NeuralNetwork::logisticLossDiff(const double& y, const double& a)
{
    return -(y/a) + (1.0 - y)/(1.0 - a);
}

/* trains network with gradient descent */
void NeuralNetwork::train(
    const std::vector<std::vector<double>>& trainingInputs,
    const std::vector<std::vector<double>>& trainingOutputs,
    const double& learningRate,
    const double& tol,
    const unsigned int& maxIter
)
{
    assert(trainingInputs.size() > 0);
    assert(trainingOutputs.size() > 0);

    // training loop
    for (unsigned int n = 0; n < maxIter; ++n)
    {
        std::vector<mathlib::Matrix> weightAdjustments = {};
        std::vector<mathlib::Matrix> biasAdjustments = {};
        for (unsigned int i = 0; i < m_weights.size(); ++i)
        {
            mathlib::Matrix dW(m_weights[i]->size(), 0.0);
            mathlib::Matrix db(m_biases[i]->size(), 0.0);
            weightAdjustments.push_back(dW);
            biasAdjustments.push_back(db);
        }

        double avgLoss = 0.0;
        
        for (unsigned int s = 0; s < trainingInputs.size(); ++s)
        {
            std::cout << "Training example: " << s << std::endl;

            assert(trainingInputs[s].size() == m_shape[0]);
            assert(trainingOutputs[s].size() == m_shape[m_numLayers - 1]);

            mathlib::Matrix trainingInput = mathlib::Matrix({1, m_shape[0]}, {trainingInputs[s]}).transpose();
            mathlib::Matrix trainingOutput = mathlib::Matrix({1, m_shape[m_numLayers - 1]}, {trainingOutputs[s]}).transpose();

            // evalulate network to get values at each layer
            mathlib::Matrix output = this->evaluate(trainingInput);

            // calculate loss and dJ/daOutputLayer
            double loss = 0.0;
            mathlib::Matrix dJ_da_outputLayer({1, m_shape[m_numLayers - 1]});
            for (unsigned int i = 0; i < m_shape[m_numLayers - 1]; ++i)
            {
                loss += logisticLoss(trainingOutput.get({i, 0}), output.get({i, 0}));
                dJ_da_outputLayer.set({0, i}, logisticLossDiff(trainingOutput.get({i, 0}), output.get({i, 0})));
            }
            avgLoss += loss / trainingInputs.size();

            std::cout << "Iteration (" << n << ") - Loss: " << loss << std::endl;
            std::cout << std::endl;
            
            // vector for chain derivative accumulation for each layer
            std::vector<mathlib::Matrix> chainDerivs(m_numLayers - 1, dJ_da_outputLayer);

            // go through each layer update weights and biases
            for (unsigned int i = m_numLayers - 1; i > 0; --i)
            {
                // for this layer calculate da/dz
                mathlib::Matrix da_dz_diag = *(m_layers[i]);
                da_dz_diag.operation(sigmoidActivationDiff);
                mathlib::Matrix da_dz = mathlib::Matrix::diag(da_dz_diag);

                // for this layer calculate dW
                mathlib::Matrix dW({m_shape[i - 1], m_shape[i]}, 0.0);
                for (unsigned int j = 0; j < m_shape[i - 1]; ++j)
                {
                    mathlib::Matrix dz_dW_byInput = mathlib::Matrix::identity(m_shape[i]) * m_layers[i - 1]->get({j, 0});
                    mathlib::Matrix dW_row = chainDerivs[i - 1] * da_dz * dz_dW_byInput;
                    dW.setRegion({j, 0}, dW_row); // set row in dW to be equal to calculated row
                }

                // for this layer calculate dz/daPrev (= weightTransposed)
                mathlib::Matrix dz_daPrev = (*(m_weights[i - 1])).transpose();

                // go through layers and update weights and biases if j = current layer or accumulate chain derivs if other layers
                for (unsigned int j = i; j > 0; --j)
                {
                    if (j == i)
                    {
                        //*(m_weights[i - 1]) += -1.0 * learningRate * dW;
                        //*(m_biases[i - 1])  += -1.0 * learningRate * (chainDerivs[i - 1] * da_dz).transpose();

                        weightAdjustments[i - 1] += dW;
                        biasAdjustments[i - 1] += (chainDerivs[i - 1] * da_dz).transpose();
                    }
                    else
                    {
                        chainDerivs[j - 1] = chainDerivs[j - 1] * da_dz * dz_daPrev;
                    }
                }
            }
        }

        /*
        mathlib::Matrix dW = weightAdjustments[0];
        mathlib::Matrix db = biasAdjustments[0];

        for (unsigned int i = 1; i < weightAdjustments.size(); ++i)
        {
            dW += weightAdjustments[i];
            db += biasAdjustments[i];
        }
        
        std::cout << "dW: " << std::endl;
        dW.display();
        std::cout << std::endl << "db: " << std::endl;
        db.display();
        std::cout << std::endl << std::endl;

        *(m_weights[i]) += -1.0 * learningRate * dW;
        *(m_biases[i])  += -1.0 * learningRate * db;
        */
        
        // make weight adjustments across all training examples
        for (unsigned int i = 0; i < m_numLayers - 1; ++i)
        {
            *(m_weights[i]) += -1.0 * learningRate * weightAdjustments[i];
            *(m_biases[i]) += -1.0 * learningRate * biasAdjustments[i];

            weightAdjustments[i].display();
            std::cout << std::endl << std::endl;
        }
        

        if (avgLoss < tol)
            break;
    }

}

/* display neural network layers and weights */
void NeuralNetwork::display()
{
    for (int i = 0; i < this->m_numLayers; ++i)
    {
        std::cout << "============== " << "LAYER " << i << " ==============" << std::endl;
        if (i > 0)
        {
            std::cout << "Weights:" << std::endl;
            this->m_weights[i - 1]->display();
            std::cout << std::endl;

            std::cout << "Biases:" << std::endl;
            this->m_biases[i - 1]->display();
            std::cout << std::endl;

            std::cout << "Prelayer:" << std::endl;
            this->m_prelayers[i - 1]->display();
            std::cout << std::endl;
        }

        std::cout << "Layer:" << std::endl;
        this->m_layers[i]->display();
        std::cout << std::endl;
    }
}