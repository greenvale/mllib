/*
Neural network library

- Uses computational graph library

- weights for each connection are indexed by layer of first node, index of first node within start layer and index of second node within destination layer
- in the computational graph, weights are input nodes (weight inputs) and neural network inputs are also inputs (static inputs)

- the total number of weight inputs is
    (layerSize_1 * layerSize_2) + (layerSize_2 * layerSize_3) + ... + (layerSize_(N-1) * layerSize_N)
*/

#pragma once

#include <compGraph.hpp>
#include <vector>
#include <assert.h>

namespace mllib
{

class NeuralNet
{
private:
    int m_numLayers;
    int m_numWeightInputs;
    int m_numStaticInputs;
    std::vector<int> m_shape;
    
    CompGraph m_nn;
    
public:
    NeuralNet();
    
    void setShape(const std::vector<int>& shape);
    std::vector<int> getShape();
    
    void construct();
    int getWeightIndex(const int& layerIndex, const int& nodeIndex1, const int& nodeIndex2);
};

} // namespace mllib

/* default ctor */
NeuralNet::NeuralNet()
{
    
}


/* get shape */ 
std::vector<int> NeuralNet::getShape()
{
    return m_shape;
}

/* set shape */
void NeuralNet::setShape(const std::vector<int>& shape)
{
    m_numLayers = shape.size();
    m_numStaticInputs = shape[0];
    
    m_numWeightInputs = 0;
    
    for (int i = 0; i < m_numLayers; ++i)
    {
        assert(shape[i] > 0); // ensure that size of each layer is positive, >0
        
        m_numWeightInputs += shape[i] * shape[i+1];
    }   
    
    m_shape = shape;
}   

/* construct */
void NeuralNet::construct()
{
    // set initial value to 0.0
    
    // construct layers of computational graph (different to layers of neural network)
    
}

/* get index of weight (layerIndex, nodeIndex1, nodeIndex2) for the input layer
- for each layer there is a matrix of size (numLayers_i, numLayers_i+1), indexed by (nodeIndex1, nodeIndex2) in row-ordered storage form
*/
int NeuralNet::getWeightIndex(const int& layerIndex, const int& nodeIndex1, const int& nodeIndex2)
{
    int index = 0;
    for (int i = 0; i < layerIndex; ++i)
    {
        index += m_shape[i] * m_shape[i+1];
    }
    index += (nodeIndex1 * m_shape[i+1]) + nodeIndex2;
    return index;
}

/* */



