/*
Linear Regression
- Requires Matrix class (available in mathlib)
*/
#pragma once

#include <matrix.hpp>
#include <assert.h>
#include <math.h>

namespace mllib
{

class LinearReg
{
private:
    int m_numDim;
    int m_numFeatures;
    Matrix<double> m_x;
    Matrix<double> m_y;
    Matrix<double> m_weights;
public:
    LinearReg();
    LinearReg(Matrix<double> x, Matrix<double> y);
    
    void addFeatures(Matrix<double> x, Matrix<double> y);
    void removeFeature(int index);
    
    void train(double alpha, double lambda, double trainTol, int maxNumIter);
    double loss(double lambda);
    Matrix<double> predict(Matrix<double> x);
};

LinearReg::LinearReg() 
{
    
}

LinearReg::LinearReg(Matrix<double> x, Matrix<double> y)
{
    assert(x.numRows() == y.numRows());
    
    m_numDim = x.numCols();
    m_numFeatures = x.numRows();
    m_x = x;
    m_y = y;
    m_x.insertCol(0, Matrix<double>(m_numFeatures, 1, 1.0)); // add bias term to x
    m_weights = Matrix<double>(m_numDim + 1, 1, 0.0); // initialise weights at zero
}

// calculate loss function given current weights and features
// lambda is the regularisation parameter
double LinearReg::loss(double lambda)
{
    assert(lambda >= 0.0);
    
    Matrix<double> result = 0.5 * (1.0/m_numFeatures) * ((m_x * m_weights) - m_y).getTranspose() * ((m_x * m_weights) - m_y) + lambda * (m_weights.getTranspose() * m_weights);
    return result.scalar();
}

// train model using gradient descent
void LinearReg::train(double alpha, double lambda, double tol, int maxNumIter)
{
    assert(alpha > 0.0);
    assert(lambda >= 0.0);
    assert(tol > 0.0);
    assert(maxNumIter > 0);
    
    m_weights = Matrix<double>(m_numDim + 1, 1, 0.0); // initialise weights at zero
    int numIterations = 0;
    for (int i = 0; i < maxNumIter; ++i)
    {
        m_weights -= alpha * (1.0/m_numFeatures) * (m_x.getTranspose()) * ((m_x * m_weights) - m_y) + (2 * lambda * m_weights); // gradient of loss function
        numIterations++;
        if (loss(lambda) <= tol)
        {
            break;
        }        
    }
    std::cout << "Final loss: " << loss(lambda) << std::endl;
    std::cout << "Number of iterations: " << numIterations << std::endl;
}

// vectorised prediction
Matrix<double> LinearReg::predict(Matrix<double> x)
{
    assert(x.numCols() == m_numDim);
    assert(x.numRows() > 0);
    
    x.insertCol(0, Matrix<double>(x.numRows(), 1, 1.0));
    Matrix<double> y = x * m_weights;
    return y;
}

}
