/*
Logistic Regression
- Requires Matrix class (available in mathlib)
*/
#pragma once

#include <matrix.hpp>
#include <assert.h>
#include <math.h>

namespace mllib
{

class LogisticReg
{
private:
    int m_numDim;
    int m_numFeatures;
    Matrix<double> m_x;
    Matrix<double> m_y;
    Matrix<double> m_weights;
public:
    LogisticReg();
    LogisticReg(Matrix<double> x, Matrix<double> y);
    
    void addFeatures(Matrix<double> x, Matrix<double> y);
    void removeFeature(int index);
    
    void train(double alpha, double lambda, double trainTol, int maxNumIter);
    double loss(double lambda);
    Matrix<double> predict(Matrix<double> x);
    
    static Matrix<double> sigmoid(Matrix<double> z);
    static double sigmoid(double z);
    static double sigmoidDeriv(double z);
};

LogisticReg::LogisticReg() 
{
    
}

LogisticReg::LogisticReg(Matrix<double> x, Matrix<double> y)
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
double LogisticReg::loss(double lambda)
{
    assert(lambda >= 0.0);
    
    double result = 0.0;
    
    for (int i = 0; i < m_numFeatures; ++i)
    {
        result += (-1.0/m_numFeatures) * ((m_y.get(i, 0) * log(sigmoid(m_x.getRow(i) * m_weights).scalar())) + (1.0 - m_y.get(i, 0)) * log(1.0 - (sigmoid(m_x.getRow(i) * m_weights).scalar())));
    }
    result += (lambda * (m_weights.getTranspose() * m_weights)).scalar(); // regularisation
    
    return result;
}

// train model using gradient descent
void LogisticReg::train(double alpha, double lambda, double tol, int maxNumIter)
{
    assert(alpha > 0.0);
    assert(lambda >= 0.0);
    assert(tol > 0.0);
    assert(maxNumIter > 0);
    
    m_weights = Matrix<double>(m_numDim + 1, 1, 0.0); // initialise weights at zero
    int numIterations = 0;
    for (int i = 0; i < maxNumIter; ++i)
    {
        m_weights -= alpha * (1.0/m_numFeatures) * (m_x.getTranspose()) * (sigmoid(m_x * m_weights) - m_y) + (2 * lambda * m_weights); // gradient of loss function
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
Matrix<double> LogisticReg::predict(Matrix<double> x)
{
    assert(x.numCols() == m_numDim);
    assert(x.numRows() > 0);
    
    x.insertCol(0, Matrix<double>(x.numRows(), 1, 1.0));
    Matrix<double> y = sigmoid(x * m_weights);
    return y;
}

// (static) vectorised sigmoid function
// copies matrix z on input
Matrix<double> LogisticReg::sigmoid(Matrix<double> z)
{
    assert(z.numCols() == 1);
    
    for (int i = 0; i < z.numRows(); ++i)
    {
        z.set(i, 0, sigmoid(z.get(i, 0)));
    }
    
    return z;
}

// (static) scalar sigmoid function
double LogisticReg::sigmoid(double z) 
{
    return 1.0 / (1.0 + exp(-z));
}

// (static) scalar sigmoid derivative function
double LogisticReg::sigmoidDeriv(double z)
{
    return sigmoid(z) * (1.0 - sigmoid(z));
}

}
