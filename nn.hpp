/*
Feed forward neural network
*/ 
#pragma once

#include <matrix.hpp>
#include <logisticRegression.hpp>
#include <vector>

namespace mllib
{

class NeuralNet
{

private:
    int m_numLayers;
    std::vector<int> m_dimensions;
    std::vector<Matrix<double>> m_nodeLayers;
    std::vector<Matrix<double>> m_preactiveLayers;
    std::vector<Matrix<double>> m_weightLayers;
public:
    NeuralNet();
    NeuralNet(std::vector<int> dimensions);
    
    void feedForwardLayer(int layerIndex);
    Matrix<double> feedForward(Matrix<double> input);
};

NeuralNet::NeuralNet()
{

}

// construct neural network with dimensions
NeuralNet::NeuralNet(std::vector<int> dimensions)
{
    m_dimensions = dimensions;
    m_numLayers = dimensions.size();
    m_nodeLayers = {};
    m_preactiveLayers = {};
    m_weightLayers = {};
    
    for (int i = 0; i < m_numLayers; ++i)
    { 
        int nodeNumRows;
        int weightNumRows;
        int weightNumCols;
        
        if (i == 0)
        {
            // input layer
            nodeNumRows = dimensions[i];
            weightNumRows = nodeNumRows;
            weightNumCols = dimensions[i + 1];
        }
        else if (i == m_numLayers - 1)
        {
            // output layer
            nodeNumRows = dimensions[i];
            weightNumRows = 0;
            weightNumCols = 0;
        }
        else
        {
            // hidden layer
            nodeNumRows = dimensions[i] + 1;
            weightNumRows = nodeNumRows;
            weightNumCols = dimensions[i + 1];
        }

        Matrix<double> nodeLayer(nodeNumRows, 1, 0.0);
        m_nodeLayers.push_back(nodeLayer);
        
        if (i > 0)
        {
            Matrix<double> preactiveLayer(nodeNumRows, 1, 0.0);
            m_preactiveLayers.push_back(preactiveLayer);
        }
        
        if ((weightNumRows > 0) && (weightNumCols > 0))
        {
            Matrix<double> weightLayer(weightNumRows, weightNumCols, 1.0);
            m_weightLayers.push_back(weightLayer);
        }
    }
}

// feed forward single layer
void NeuralNet::feedForwardLayer(int layerIndex)
{
    assert(layerIndex < m_numLayers - 1);
    
    // calculate preactivated layer
    Matrix<double> preactive = m_weightLayers[layerIndex].getTranspose() * m_nodeLayers[layerIndex];
    if (layerIndex < m_numLayers - 2)
    {
        // hidden layer
        m_preactiveLayers[layerIndex].setRegion(1, m_dimensions[layerIndex + 1], 0, 0, preactive);
    }
    else
    {
        // output layer
        m_preactiveLayers[layerIndex].setRegion(0, m_dimensions[layerIndex + 1] - 1, 0, 0, preactive);
    }
    
    // activate layer
    m_nodeLayers[layerIndex + 1] = mllib::LogisticReg::sigmoid(m_preactiveLayers[layerIndex]);
}

// feed forward whole NN
Matrix<double> NeuralNet::feedForward(Matrix<double> input)
{
    assert(input.numRows() == m_nodeLayers[0].numRows());
    assert(input.numCols() == 1);
    
    m_nodeLayers[0] = input;
    for (int i = 0; i < m_numLayers - 1; ++i)
    {
        feedForwardLayer(i);
    }
    
    return m_nodeLayers[m_numLayers - 1];
}

// 


}
