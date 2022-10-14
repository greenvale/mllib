/*
Computational Graph Library V2
- More functional than V1 - node class structure is not used
*/

#pragma once

#include <iostream>
#include <vector>
#include <stack>
#include <assert.h>
#include <algorithm>
#include <math.h>

/* Node type enumerator */
enum NodeType { INPUT, OUTPUT, STEP };

/* Operator base class */
class Operator
{
private:
public:
    virtual void set2identity(double& val)
    {
    }
    virtual void exec(double& childVal, const double& parentVal)
    {
    }
    virtual double deriv(const double& parentVal, const double& childVal) 
    {
        // output is operator execution on input with potentially other inputs
        // deriv should only depend on output and input differentiation is with respect to
        return 0.0;
    }
};


/* Constant operator */
class Constant : public Operator
{
private:
public:
};

/* Sum operator */
class Sum : public Operator
{
private:
public:
    void set2identity(double& val)
    {
        val = 0.0;
    }
    void exec(double& childVal, const double& parentVal)
    {
        childVal += parentVal;
    }
    double deriv(const double& parentVal, const double& childVal)
    {
        return 1.0;
    }
};

/* Multiply operator */
class Mult : public Operator
{
private:
public:
    void set2identity(double& val)
    {
        val = 1.0;
    }
    void exec(double& childVal, const double& parentVal)
    {
        childVal *= parentVal;
    }
    double deriv(const double& parentVal, const double& childVal)
    {
        return childVal / parentVal;
    }
};

void printCoord(std::vector<int> vec) 
{
    std::cout << "{" << vec[0] << ", " << vec[1] << "}" << std::endl;
}

/* Power operator */
class Power : public Operator
{
private:
    double m_exp;
public:
    Power(const double& exp)
    {
        m_exp = exp;
    }
    void set2identity(double& val)
    {
        val = 1.0;
    }
    void exec(double& childVal, const int& parentVal)
    {
        childVal *= pow(parentVal, m_exp);
    }
    double deriv(const double& parentVal, const double& childVal)
    {
        return m_exp * childVal / parentVal;
    }
};

/* ************************************************************************************ */

class CompGraph 
{

private:
    std::vector<int> m_shape;
    
    std::vector<std::vector< double >> m_vals;
    std::vector<std::vector< Operator* >> m_operatorPtrs;
    std::vector<std::vector< std::vector<std::vector<int>> >> m_parentCoords;
        
public:
    CompGraph();
    CompGraph(const std::vector<int>& shape);
    
    // graph construction
    void set(const std::vector<int>& coord, Operator* op, const NodeType& type);
    void join(const std::vector<std::vector<int>>& childCoords, const std::vector<std::vector<int>>& parentCoords);
    
    // graph operation
    std::vector<double> read(const std::vector<std::vector<int>>& coords);
    void write(const std::vector<std::vector<int>>& coords, const std::vector<double> vals);
    void exec();
    
    // graph optimisation
    std::vector<std::vector<int>> derivChain(const std::vector<int>& denomCoord, const std::vector<int>& numerCoord);
    double deriv(const std::vector<std::vector<int>>& chain);
    std::vector<double> gradDescent(
        const std::vector<std::vector<int>>& optimInputCoords,
        const std::vector<std::vector<int>>& staticInputCoords,
        const std::vector<int>& costCoord, // single scalar output
        const double& alpha,
        const double& tol,
        const int& maxIter,
        const std::vector<double>& optimInputInit,
        const std::vector<std::vector<double>>& staticInputBatch
    );
};

/* default ctor */
CompGraph::CompGraph()
{

}

/* ctor with shape and empty node entries */
CompGraph::CompGraph(const std::vector<int>& shape)
{
    this->m_shape = shape;
    
    for (int i = 0; i < this->m_shape.size(); ++i)
    {
        this->m_vals.push_back(std::vector< double >(this->m_shape[i]));
        this->m_operatorPtrs.push_back(std::vector< Operator* >(this->m_shape[i]));
        this->m_parentCoords.push_back(std::vector< std::vector<std::vector<int>> >(this->m_shape[i]));
    }
}

/* set the operator and type of a node at given coordinate */
void CompGraph::set(const std::vector<int>& coord, Operator* opPtr, const NodeType& type)
{
    // check that coord is valid
    assert(coord[0] >= 0);
    assert(coord[0] < this->m_shape.size());
    assert(coord[1] >= 0);
    assert(coord[1] < this->m_shape[coord[0]]);
    
    this->m_operatorPtrs[coord[0]][coord[1]] = opPtr;
}

/* join a list of parent nodes with a list of child nodes */
void CompGraph::join(const std::vector<std::vector<int>>& parentCoords, const std::vector<std::vector<int>>& childCoords)
{
    for (int i = 0; i < childCoords.size(); ++i)
    {
        for (int j = 0; j < parentCoords.size(); ++j)
        {
            // check that coord is valid
            assert(childCoords[i][0] >= 0);
            assert(childCoords[i][0] < this->m_shape.size());
            assert(childCoords[i][1] >= 0);
            assert(childCoords[i][1] < this->m_shape[childCoords[i][0]]);
            // check that coord is valid
            assert(parentCoords[j][0] >= 0);
            assert(parentCoords[j][0] < this->m_shape.size());
            assert(parentCoords[j][1] >= 0);
            assert(parentCoords[j][1] < this->m_shape[parentCoords[j][0]]);
            
            this->m_parentCoords[childCoords[i][0]][childCoords[i][1]].push_back(parentCoords[j]);
        }
    }
}

/* read from node list */
std::vector<double> CompGraph::read(const std::vector<std::vector<int>>& coords)
{
    std::vector<double> vals(coords.size());
    for (int i = 0; i < coords.size(); ++i)
    {
        vals[i] = this->m_vals[coords[i][0]][coords[i][1]];
    }
    return vals;
}

/* write to node list */
void CompGraph::write(const std::vector<std::vector<int>>& coords, const std::vector<double> vals)
{
    for (int i = 0; i < coords.size(); ++i)
    {
        this->m_vals[coords[i][0]][coords[i][1]] = vals[i];
    }
}

/* execute computational graph 
*/
void CompGraph::exec()
{
    for (int i = 0; i < this->m_shape.size(); ++i)
    {
        for (int j = 0; j < this->m_shape[i]; ++j)
        {
            // set val to identity
            this->m_operatorPtrs[i][j]->set2identity(this->m_vals[i][j]);
            
            // run operation with respect to parent values
            for (int k = 0; k < this->m_parentCoords[i][j].size(); ++k)
            {
                std::vector<int> parentCoord = this->m_parentCoords[i][j][k];
                this->m_operatorPtrs[i][j]->exec(this->m_vals[i][j], this->m_vals[parentCoord[0]][parentCoord[1]]);
            }
        }
    }
}

/* ************************************************************************************ */
// OPTIMISATION

/* get derivative chain for TREE-STRUCTURED graph between numerator node and denominator node
*/
std::vector<std::vector<int>> CompGraph::derivChain(const std::vector<int>& denomCoord, const std::vector<int>& numerCoord)
{
    // check that coord is valid
    assert(denomCoord[0] >= 0);
    assert(denomCoord[0] < this->m_shape.size());
    assert(denomCoord[1] >= 0);
    assert(denomCoord[1] < this->m_shape[denomCoord[0]]);
    // check that coord is valid
    assert(numerCoord[0] >= 0);
    assert(numerCoord[0] < this->m_shape.size());
    assert(numerCoord[1] >= 0);
    assert(numerCoord[1] < this->m_shape[numerCoord[0]]);

    std::vector<std::vector<int>> chain = {};
    std::stack< std::vector<std::vector<int>> > chainStack;
    
    chainStack.push( { numerCoord } ); // initialise chain stack with numerator coordinate
    bool stopFlag = false;
    while (stopFlag == false)
    {
        // pop the stack 
        std::vector<std::vector<int>> chainSection = chainStack.top();
        chainStack.pop();
        
        // get child nodes if this isn't the denom ind
        std::vector<int> chainDenomCoord = chainSection[chainSection.size() - 1];
        if (chainDenomCoord == denomCoord)
        {
            // break
            chain = chainSection;
            stopFlag = true;
        }
        else
        {
            // get parent nodes and add to chain
            std::vector<std::vector<int>> chainDenomParentCoords = this->m_parentCoords[chainDenomCoord[0]][chainDenomCoord[1]];
            for (int i = 0; i < chainDenomParentCoords.size(); ++i)
            {   
                std::vector<std::vector<int>> newChainSection = chainSection;
                newChainSection.push_back(chainDenomParentCoords[i]);
                chainStack.push(newChainSection);
            }
        }
        
        if ((chainStack.size() == 0) && (stopFlag != true))
        {
            return {{}}; // if at end of stack and have not already finished then derivative chain not found
        }
    }
    
    std::reverse(chain.begin(), chain.end());
    return chain;
}   

/* get derivative given a derivative chain using chain rule
    comp graph should be executed before calculating derivative
*/
double CompGraph::deriv(const std::vector<std::vector<int>>& chain)
{
    double d = 1.0; // set to multiplicative identity
    for (int i = 0; i < chain.size() - 1; ++i)
    {
        d *= this->m_operatorPtrs[chain[i+1][0]][chain[i+1][1]]->deriv( // using numerator operation
            this->m_vals[chain[i][0]][chain[i][1]],   // denominator (input)
            this->m_vals[chain[i+1][0]][chain[i+1][1]] // numerator (output)
        ); 
    }    
    return d;
}

/* gradient descent with respect to output - must be a tree graph! */
std::vector<double> CompGraph::gradDescent(
    const std::vector<std::vector<int>>& optimInputCoords,
    const std::vector<std::vector<int>>& staticInputCoords,
    const std::vector<int>& costCoord, // single scalar output
    const double& alpha,
    const double& tol,
    const int& maxIter,
    const std::vector<double>& optimInputInit,
    const std::vector<std::vector<double>>& staticInputBatch
)
{
    // housekeeping
    int numOptimInputs = optimInputCoords.size();
    int numStaticInputs = staticInputCoords.size();
    int batchSize = staticInputBatch.size();  
    
    // get deriv chains for each optimisation input
    std::vector<std::vector<std::vector<int>>> optimInputDerivChains(numOptimInputs);
    std::vector<double> optimInputDerivs(numOptimInputs);
    for (int i = 0; i < numOptimInputs; ++i)
    {
        optimInputDerivChains[i] = this->derivChain(optimInputCoords[i], costCoord);
    }
    
    // initialise inputs (optimisation and static)
    std::vector<double> optimInputs = optimInputInit; // store updated optimised inputs, initially set to initial vals
    
    
    // optimisation iterative loop
    bool stopFlag = false;
    int iter = 0;
    while (stopFlag == false)
    {
        // set derivatives to zero
        for (int i = 0; i < numOptimInputs; ++i)
        {
            optimInputDerivs[i] = 0.0;
        }
    
        // loop through batch of data to get derivatives
        for (int i = 0; i < batchSize; ++i)
        {
            // set static inputs with sample from batch
            for (int j = 0; j < numStaticInputs; ++j)
            {
                this->m_vals[staticInputCoords[j][0]][staticInputCoords[j][1]] = staticInputBatch[i][j];
            }
            
            // set optimisation inputs with existing copy
            for (int j = 0; j < numOptimInputs; ++j)
            {
                this->m_vals[optimInputCoords[j][0]][optimInputCoords[j][1]] = optimInputs[j];
            }
            
            // execute graph
            this->exec();
            
            /*
            for (int j = 0; j < m_shape.size(); ++j)
            {
                for (int k = 0; k < m_shape[j]; ++k)
                {
                    std::cout << this->m_vals[j][k] << std::endl;
                }
            } */
            
            // get cost
            double cost = this->m_vals[costCoord[0]][costCoord[1]];
            std::cout << "Cost: " << cost << std::endl;
            
            // get derivatives of cost with respect to optimisation inputs
            for (int j = 0; j < numOptimInputs; ++j)
            {
                optimInputDerivs[j] += (1.0 / batchSize) * this->deriv(optimInputDerivChains[j]);
            }
        }
        
        // adjust optimisation inputs
        for (int i = 0; i < numOptimInputs; ++i)
        {
            optimInputs[i] += -1.0 * alpha * optimInputDerivs[i];
        }
        
        // increment counter
        iter++;
        
        // calculate size of optimInputDerivs
        double optimInputDerivsNorm = 0.0;
        for (int i = 0; i < numOptimInputs; ++i)
        {
            optimInputDerivsNorm += optimInputDerivs[i] * optimInputDerivs[i];
            std::cout << optimInputs[i] << std::endl;
        }
        std::cout << "Iteration: " << iter - 1 << "; norm: " << optimInputDerivsNorm << std::endl;
        
        // if converged on a final value or exceeded max iterations then stop
        if ((optimInputDerivsNorm < tol) || (iter >= maxIter))
        {
            stopFlag = true;
        }
    }
    
    return optimInputs;
}

