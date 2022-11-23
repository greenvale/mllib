/* Computational graph library, W Denny
    - computational graph's for machine learning
    - includes optimisation functions targetted for machine learning
    - note there is NO SAFEGUARDING so far in this class, that means any incorrect implementation cannot be prevented
*/

#pragma once
#include <vector>
#include <numeric>
#include <iostream>
#include <cmath>

namespace mllib
{

/* Position data structure 
    - graphs are represented in 2d, nodes have a row and a col
*/
class Pos
{
public:
    unsigned int m_col;
    unsigned int m_row;
    Pos() {}
    Pos(const unsigned int& col, const unsigned int& row)
    {
        this->m_col = col;
        this->m_row = row;
    }
};

/*****************************************************************************************************/

/* Node data structure
    - holds a value in double precision
    - holds a pointer to an operation base class functor that in reality is a derived class
    - holds a position value in the graph
    - holds array of ptrs to child nodes
*/
class Op; // declare op class to reference ptr to Op
class Node
{
public:
    Pos m_pos;
    double m_val = 0.0;
    Op* m_op;
    std::vector<Node*> m_parArr;
    std::vector<Node*> m_childArr;
    std::vector<double> m_derivArr;
    Node() {}
};

/*****************************************************************************************************/

/* Operation functors
    - contains operator() function that takes a node ptr and executes a mathematical function using the values from parent nodes of the given nodes
    - the derivatives function is to be run once the operator() function is executed
    - the derivatives function calculates the derivatives with respect to each of the parent nodes and stores them in derivArr in the node
*/
/* base class */
class Op
{
public:
    Op() {}
    Op(const double& val) {}
    virtual void operator()(Node* node) { }
    virtual void derivatives(Node* node) { } // takes derivative with respect to parent ptr
};

/* summation 
    - takes any number of inputs
*/
class Sum : public Op
{
public:
    void operator()(Node* node)
    {
        node->m_val = 0.0; // set to identity
        for (int i = 0; i < node->m_parArr.size(); ++i)
        {
            node->m_val += node->m_parArr[i]->m_val;
        }
    }
    void derivatives(Node* node)
    {
        for (int i = 0; i < node->m_derivArr.size(); ++i)
        {
            node->m_derivArr[i] = 1.0;
        }
    }
};

/* multiplication 
    - takes any number of inputs
*/
class Mul : public Op
{
public:
    void operator()(Node* node)
    {
        node->m_val = 1.0; // set to identity
        for (int i = 0; i < node->m_parArr.size(); ++i)
        {
            node->m_val *= node->m_parArr[i]->m_val;
        }
    }
    void derivatives(Node* node)
    {
        for (int i = 0; i < node->m_parArr.size(); ++i)
        {
            node->m_derivArr[i] = node->m_val / node->m_parArr[i]->m_val;
        }
    }
};

/* difference 
    - takes exactly two inputs
    - assumes exactly two parents in parArr
    - 0th parent val - 1st parent val
*/
class Dif : public Op
{
public:
    void operator()(Node* node)
    {
        node->m_val = node->m_parArr[0]->m_val - node->m_parArr[1]->m_val;
    }
    void derivatives(Node* node)
    {
        node->m_derivArr[0] = 1.0;
        node->m_derivArr[1] = -1.0;
    }
}; 

/* square
    - takes exactly one input
    - assumes only one parent in parArr
*/
class Squ : public Op
{
public:
    void operator()(Node* node)
    {
        node->m_val = node->m_parArr[0]->m_val * node->m_parArr[0]->m_val;
    }
    void derivatives(Node* node)
    {
        node->m_derivArr[0] = 2.0 * node->m_parArr[0]->m_val;
    }
};

/* sigmoid
    - takes exactly one input
    - assumes only one parent in parArr
    - derivatives must only be executed once operator() has been executed
*/
class Sig : public Op
{
public:
    void operator()(Node* node)
    {
        node->m_val = exp(-1.0 * node->m_parArr[0]->m_val);
        node->m_val = 1.0 / (1.0 + node->m_val);
        //std::cout << "Sigmoid val: " << node->m_val << std::endl;
    }
    void derivatives(Node* node)
    {
        node->m_derivArr[0] = node->m_val * (1.0 - node->m_val);
        //std::cout << "Sigmoid derivative: " << node->m_derivArr[0] << std::endl;
    }
};

/*****************************************************************************************************/

/* Config for computational graph - element of adjacency list
    - adjacency list of each node with its children in position form
*/
class AdjListElem
{
public:
    Pos m_pos;
    std::vector<Pos> m_parArr;
    std::vector<Pos> m_childArr;
    Op* m_op;
    AdjListElem() {}
    AdjListElem(const Pos& pos, const std::vector<Pos>& parArr, const std::vector<Pos>& childArr, Op* op)
    {
        this->m_pos = pos;
        this->m_parArr = parArr;
        this->m_childArr = childArr;
        this->m_op = op;
    }
};

/*****************************************************************************************************/

/* Chain of derivatives

*/
class DerivChain
{
public:
    std::vector<Node*> m_nodeArr;
    std::vector<unsigned int> m_indArr;
    DerivChain()
    {

    }
};

/*****************************************************************************************************/

/* Computational graph class 
    - exec operation executes each col to get node value and calculates the derviatives
    - derivatives with respect to each node's parent are calculated and stored in an array in the node
*/
class CompGraph
{   
private:
    unsigned int m_numNodes;
    std::vector<unsigned int> m_shape;
    std::vector<Node*> m_nodeArr;
public:
    CompGraph() = delete;
    CompGraph(const std::vector<unsigned int>& shape, const std::vector<AdjListElem*>& adjList);
    unsigned int pos2ind(const Pos& pos);
    void exec();
    double readVal(const Pos& pos);
    double readDeriv(const Pos& pos, const unsigned int& ind);
    void writeVal(const Pos& pos, const double& val);
    void reset();

    // graph union
    void append(const CompGraph& cg); // fully connects another comp graph

    // optimisation
    DerivChain getChain(const Pos& start, const Pos& end);
    double chainDeriv(const DerivChain& chain);
    void optimise(
        const std::vector<Pos>& weightPosArr,
        const std::vector<Pos>& staticPosArr,
        const Pos& costPos,
        const std::vector<double>& initWeight,
        const std::vector<std::vector<std::vector<double>>>& batchArray
    );
};

/* ctor */
CompGraph::CompGraph(const std::vector<unsigned int>& shape, const std::vector<AdjListElem*>& adjList)
{
    this->m_shape = shape; // copy shape vector
    this->m_numNodes = std::accumulate(this->m_shape.begin(), this->m_shape.end(), 0); // get total number of nodes
    this->m_nodeArr = std::vector<Node*>(this->m_numNodes); // initialise node ptr array

    // create nodes
    for (int i = 0; i < adjList.size(); ++i)
    {
        this->m_nodeArr[i] = new Node();
        this->m_nodeArr[i]->m_op = adjList[i]->m_op;
        this->m_nodeArr[i]->m_pos = adjList[i]->m_pos;
        this->m_nodeArr[i]->m_derivArr = std::vector<double>(adjList[i]->m_parArr.size(), 0.0); // derivative array is same size as parent array
    }

    // link nodes in the graph
    for (int i = 0; i < adjList.size(); ++i)
    {
        this->m_nodeArr[i]->m_parArr = {};
        this->m_nodeArr[i]->m_childArr = {};
        for (int j = 0; j < adjList[i]->m_parArr.size(); ++j)
        {
            Node* par = this->m_nodeArr[this->pos2ind(adjList[i]->m_parArr[j])];
            this->m_nodeArr[i]->m_parArr.push_back(par);
        }
        for (int j = 0; j < adjList[i]->m_childArr.size(); ++j)
        {
            Node* child = this->m_nodeArr[this->pos2ind(adjList[i]->m_childArr[j])];
            this->m_nodeArr[i]->m_childArr.push_back(child);
        }
    }
}

/* position converted to index in nodeArr 
     - note this function requires simple optimisation by using accumulations
*/
unsigned int CompGraph::pos2ind(const Pos& pos)
{
    unsigned int ind = 0;
    for (int i = 0; i < pos.m_col; ++i)
        ind += this->m_shape[i];
    ind += pos.m_row;
    return ind;
}

/* read from node value */
double CompGraph::readVal(const Pos& pos) 
{
    return this->m_nodeArr[this->pos2ind(pos)]->m_val;
}

/* reads from deriv array at given index within array */
double CompGraph::readDeriv(const Pos& pos, const unsigned int& ind)
{
    return this->m_nodeArr[this->pos2ind(pos)]->m_derivArr[ind];
}

/* write to node value */
void CompGraph::writeVal(const Pos& pos, const double& val) 
{
    this->m_nodeArr[this->pos2ind(pos)]->m_val = val;
}

/* resets the graph by setting all values to zero */
void CompGraph::reset() 
{
    for (int i = 0; i < this->m_numNodes; ++i)
    {
        this->m_nodeArr[i]->m_val = 0.0; // reset node values
        for (int j = 0; j < this->m_nodeArr[i]->m_derivArr.size(); ++j)
        {
            this->m_nodeArr[i]->m_derivArr[j] = 0.0; // reset value of derivative of node with respect to each child
        }
    }
}

/* execute graph */ 
void CompGraph::exec()
{
    for (int i = 0; i < this->m_numNodes; ++i)
    {
        if (this->m_nodeArr[i]->m_op != nullptr) // check if operation has been defined
        {
            (*this->m_nodeArr[i]->m_op)(this->m_nodeArr[i]); // calculate values
            this->m_nodeArr[i]->m_op->derivatives(this->m_nodeArr[i]); // calculate derivatives
        }
    }
}

/* append another graph to this graph */
void CompGraph::append(const CompGraph& cg)
{

}

/*****************************************************************************************************/
/* Optimisation */

/* get chain between two nodes
    - uses a queue for breadth-first search
    - goes in reverse from end node to start node, because indexes apply to the child node
 */
DerivChain CompGraph::getChain(const Pos& start, const Pos& end)
{
    Node* startNode = this->m_nodeArr[this->pos2ind(start)];
    Node* endNode = this->m_nodeArr[this->pos2ind(end)];

    std::vector<DerivChain> queue = {};

    // initialise queue with singleton deriv chains onto the parent nodes for the end node
    for (unsigned int i = 0; i < endNode->m_parArr.size(); ++i)
    {
        DerivChain dc;
        dc.m_indArr = { i }; // add parent index from child node
        dc.m_nodeArr = { endNode }; // add child node
        queue.push_back(dc);
    }

    // dequeue and process nodes until the queue is empty or deriv chain found
    while (queue.size() > 0)
    {
        DerivChain dc = queue[0]; // copy deriv chain from queue and dequeue
        queue.erase(queue.begin());
        
        // check if this is the necessary deriv chain
        if ((dc.m_nodeArr.back())->m_parArr[dc.m_indArr.back()] == startNode)
        {
            return dc;
        }
        else // the node that is pointed to by the index in the child's parent array is not the start node
        {
            for (unsigned int i = 0; i < (dc.m_nodeArr.back())->m_parArr[dc.m_indArr.back()]->m_parArr.size(); ++i)
            {
                DerivChain dcNew = dc; // copy dc and add extra node
                dcNew.m_indArr.push_back( i );
                dcNew.m_nodeArr.push_back( (dc.m_nodeArr.back())->m_parArr[dc.m_indArr.back()] );
                queue.push_back(dcNew);
            }
        }
    }
    return DerivChain(); // return empty deriv chain as chain not found
}

/* calculate chain derivative */
double CompGraph::chainDeriv(const DerivChain& chain)
{
    double result = 1.0; // set to multiplicative identity
    for (int i = 0; i < chain.m_indArr.size(); ++i)
    {
        result *= chain.m_nodeArr[i]->m_derivArr[chain.m_indArr[i]];
    }
    return result;
}

/* optimise for a vector of batches of sample data
    - currently automatically creates chain derivs
    - input variables are either weight or static
    - the sample data is static
    - the optimisation parameters are weights
    - gradient descent is used
    - optimised such that some scalar cost is zero
*/
void CompGraph::optimise(
    const std::vector<Pos>& weightPosArr,
    const std::vector<Pos>& staticPosArr,
    const Pos& costPos,
    const std::vector<double>& initWeight,
    const std::vector<std::vector<std::vector<double>>>& batchArr
)
{
    // get derivative chains
    std::vector<DerivChain> derivChainArr;
    for (int i = 0; i < weightPosArr.size(); ++i)
    {
        derivChainArr.push_back(this->getChain(weightPosArr[i], costPos));
    }

    std::vector<double> derivArr(weightPosArr.size()); // allocate derivative array

    // initialise weights
    for (int i = 0; i < weightPosArr.size(); ++i)
    {
        this->m_nodeArr[this->pos2ind(weightPosArr[i])]->m_val = initWeight[i];
    }

    bool stop = false;
    int counter = 0;
    while (stop == false)
    {
        std::cout << "Iteration: " << counter << std::endl;
        int batchInd = counter % batchArr.size();
        std::cout << "Batch index: " << batchInd << std::endl;
        double derivTot = 0.0;

        // set the deriv array elements to zero - this will be used for accumulating error
        for (int i = 0; i < derivArr.size(); ++i)
            derivArr[i] = 0.0;

        // loop through samples in batch and accumulate cost derivatives for each weight in derivArr
        for (int i = 0; i < batchArr[batchInd].size(); ++i)
        {
            // set sample value
            for (int j = 0; j < staticPosArr.size(); ++j)
            {
                this->m_nodeArr[this->pos2ind(staticPosArr[j])]->m_val = batchArr[batchInd][i][j];
            }

            // execute graph
            this->exec();

            // calculate chain derivs of cost with respect to each weight
            for (int j = 0; j < derivArr.size(); ++j)
            {
                derivArr[j] += this->chainDeriv(derivChainArr[j]);
                derivTot += this->chainDeriv(derivChainArr[j]);
            }
        }

        std::cout << "Cost: " << this->m_nodeArr[this->pos2ind(costPos)]->m_val << std::endl;
        // adjust weights
        for (int i = 0; i < weightPosArr.size(); ++i)
        {
            this->m_nodeArr[this->pos2ind(weightPosArr[i])]->m_val += -0.5 * 0.01 * derivArr[i];
            std::cout << "Derivative for weight " << i << ": " << derivArr[i] << std::endl;
        }

        // check convergence
        if (derivTot < 0.001)
        {
            stop = true;
        }
        counter++;
    }
}



/**********************************************************************************************************************************************/
/* DEMO GRAPHS */

CompGraph* ANDGate()
{
    Sum* sum = new Sum;
    Mul* mul = new Mul;
    Squ* squ = new Squ;
    Dif* dif = new Dif;
    Sig* sig = new Sig;

    // nodes adjacency list
    std::vector<AdjListElem*> adjList;
    // col 0
    adjList.push_back(new AdjListElem(Pos(0, 0), {}, {Pos(1, 0)}, nullptr)); // weight 0
    adjList.push_back(new AdjListElem(Pos(0, 1), {}, {Pos(1, 1)}, nullptr)); // weight 1
    adjList.push_back(new AdjListElem(Pos(0, 2), {}, {Pos(1, 0)}, nullptr)); // imput 0
    adjList.push_back(new AdjListElem(Pos(0, 3), {}, {Pos(1, 1)}, nullptr)); // input 1
    // col 1
    adjList.push_back(new AdjListElem(Pos(1, 0), {Pos(0, 0), Pos(0, 2)}, {Pos(2, 0)}, mul));
    adjList.push_back(new AdjListElem(Pos(1, 1), {Pos(0, 1), Pos(0, 3)}, {Pos(2, 0)}, mul));
    // col 2
    adjList.push_back(new AdjListElem(Pos(2, 0), {Pos(1, 0), Pos(1, 1)}, {Pos(3, 0)}, sum));
    // col 3
    adjList.push_back(new AdjListElem(Pos(3, 0), {Pos(2, 0)}, {Pos(4, 0)}, sig)); // sigmoid 
    adjList.push_back(new AdjListElem(Pos(3, 1), {}, {Pos(4, 0)}, nullptr)); // correct input
    // col 4
    adjList.push_back(new AdjListElem(Pos(4, 0), {Pos(3, 0), Pos(3, 1)}, {Pos(5, 0)}, dif));
    // col 5
    adjList.push_back(new AdjListElem(Pos(5, 0), {Pos(4, 0)}, {}, squ)); // cost

    // create graph
    CompGraph* cg = new CompGraph({4, 2, 1, 2, 1, 1}, adjList);
    
    return cg;
}

}; // namespace mllib
