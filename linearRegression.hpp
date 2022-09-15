/*
Linear Regression
*/

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
        void train(double alpha, double trainTol, int maxNumIter);
        double cost();
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

// calculate cost function given current weights and features
double LinearReg::cost()
{
    Matrix<double> result = 0.5 * (1.0/m_numFeatures) * ((m_x * m_weights) - m_y).getTranspose() * ((m_x * m_weights) - m_y);
    return result.get(0,0);
}

// train model using gradient descent
void LinearReg::train(double alpha, double tol, int maxNumIter)
{
    assert(alpha > 0.0);
    assert(tol > 0.0);
    assert(maxNumIter > 0);
    
    m_weights = Matrix<double>(m_numDim + 1, 1, 0.0); // initialise weights at zero
    //double prevCost = cost();
    int numIterations = 0;
    for (int n = 0; n < maxNumIter; ++n)
    {
        m_weights -= alpha * (1.0/m_numFeatures) * (m_x.getTranspose()) * ((m_x * m_weights) - m_y);
        //double newCost = cost();
        //double costDif = fabs(newCost - prevCost);
        //prevCost = newCost;
        numIterations++;
        if (cost() <= tol)
        {
            break;
        }        
    }
    std::cout << "Final cost: " << cost() << std::endl;
    std::cout << "Number of iterations: " << numIterations << std::endl;
}

// vectorised prediction
Matrix<double> LinearReg::predict(Matrix<double> x)
{
    x.insertCol(0, Matrix<double>(x.numRows(), 1, 1.0));
    Matrix<double> y = x * m_weights;
    return y;
}

}
