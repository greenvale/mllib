/*
Computational Graph library
- feed-forward, i.e. you have inputs are computed with operations to get outputs in one direction
- double precision
- nodes are singly linked
- does not work with cycles
*/

#include <vector>
#include <assert.h>

/*
***************************************************************************************************************************
NODE
- contains value
- contains vector of parent nodes on which node value will depend
*/

class Node
{
protected:
    double m_value;
    int m_numParents;
    std::vector<Node*> m_parents;
public:
    Node();
    double value() const;
    int findParent(const Node* n) const;
    virtual void addParent(Node* n); // can be overriden for output node
    void removeParent(const Node* n);
    
    // overriden methods
    virtual void exec() {}
    virtual double deriv(const Node* n) const { return 0.0; } // gets derivative with respect to other node (zero by default, must be specified in derived class)
};

/* ctor */
Node::Node()
{
    this->m_value = 0.0;
    this->m_numParents = 0;
    this->m_parents = {};
}

/* returns value of node */
double Node::value() const
{
    return this->m_value;
}

/* find output index by node ptr, returns -1 if not found */
int Node::findParent(const Node* n) const
{
    assert(n != nullptr);
    for (int i = 0; i < this->m_numParents; ++i)
    {
        if (this->m_parents[i] == n)
        {
            return i;
        }
    }
    return -1;
}

/* add parent by node ptr */
void Node::addParent(Node* n)
{
    assert(this != n); // nodes cannot connect to themselves
    assert(this->findParent(n) == -1); // make sure output isn't already present [O(n) search]
    this->m_parents.push_back(n);
    this->m_numParents++;
}

/* remove parent by node ptr */
void Node::removeParent(const Node* n)
{
    int ind = this->findParent(n); // [O(n) search]
    assert(ind != -1); // make sure input is present
    this->m_parents.erase(this->m_parents.begin() + ind);
    this->m_numParents--;
}

/*
***************************************************************************************************************************
INPUT
- no derivative as it is not applicable
- no exec as it is not applicable
*/

class Input : public Node
{
private:
public:
    void set(const double& value);
};

void Input::set(const double& value)
{
    this->m_value = value;
}

/*
***************************************************************************************************************************
OUTPUT
- contains value of one other operation
*/

class Output : public Node
{
private:
public:
    void addParent(Node* n);
    void exec();
    double deriv(const Node* n) const;
};

/* add parent - overwritten to ensure only one parent added */
void Output::addParent(Node* n)
{
    assert(this->m_numParents == 0); // output can only be connected to the value of one node
    
    assert(this->findParent(n) == -1); // make sure output isn't already present
    this->m_parents.push_back(n);
    this->m_numParents++;
    
}

/* execute */
void Output::exec()
{
    this->m_value = this->m_parents[0]->value();
}

/* derivative 
- always constant 1
*/
double Output::deriv(const Node* n) const
{
    return 1.0;
}

/*
***************************************************************************************************************************
SUM
- exec sums input values into value
*/ 

class Sum : public Node
{
private:
public:
    void exec();
    double deriv(const Node* n) const;
};

/* execute */
void Sum::exec()
{
    assert(this->m_numParents > 0); // ensure that operation is valid
    this->m_value = 0.0; // set to additive identity
    for (int i = 0; i < this->m_numParents; ++i)
    {
        this->m_value += this->m_parents[i]->value();
    }
}

/* derivative 
- always constant 1
*/
double Sum::deriv(const Node* n) const
{
    if (this->findParent(n) != -1)
    {
        return 1.0;
    }
    else
    {
        return 0.0;
    }
}

/*
***************************************************************************************************************************
MULTIPLY
- exec multiplies input values into value
*/ 

class Mult : public Node
{
private:
public:
    void exec();
    double deriv(const Node* n) const;
};

/* execute */
void Mult::exec()
{
    assert(this->m_numParents > 0); // ensure that operation is valid
    this->m_value = 1.0; // set to multiplicative identity
    for (int i = 0; i < this->m_numParents; ++i)
    {
        this->m_value *= this->m_parents[i]->value();
    }
}

/* derivative 
- differentiates operator with respect to input (multiple of other inputs values)
- does not use assert to avoid extra O(n) search
*/
double Mult::deriv(const Node* n) const
{
    double flag = 0.0;
    double d = 1.0; // set to multiplicative identity
    for (int i = 0; i < this->m_numParents; ++i)
    {
        if (this->m_parents[i] != n)
        {
            d *= this->m_parents[i]->value();
        }
        else
        {
            flag = 1.0; // n is a parent node, therefore raise flag to 1.0 to not return 0.0
        }
    }
    return d * flag;
}

/*
***************************************************************************************************************************
COMPUTATIONAL GRAPH
- each layer contains nodes that are independent of eachother meaning they can be executed in any order
- uses (layerInd, nodeInd) indexing method instead of ptrs
*/ 

class CompGraph
{
private:
    int m_numLayers;
    int m_numInputs;
    int m_numOutputs;
    std::vector<int> m_shape;
    std::vector<std::vector<Node*>> m_layers;   // vector of vectors of node ptrs for each layer
    std::vector<Input*> m_inputs;                  // vector of node ptrs that are inputs to graph
    std::vector<Output*> m_outputs;                // vector of node ptrs that are outputs of graph
    
    //std::vector<Node*> m_staticInputs;          // for graph optimisation against cost function           
    //std::vector<Node*> m_optimInputs;
    
public:
    CompGraph();
    CompGraph(const std::vector<int>& shape);
    
    void set(const std::vector<int>& ind, Node* n);                    // sets node ptr at (layerInd, nodeInd)
    void setInput(const std::vector<int>& ind, Input* n);
    void setOutput(const std::vector<int>& ind, Output* n);
    Node* get(const std::vector<int>& ind) const;       // gets node ptr at (layerInd, nodeInd)
    
    void join(const std::vector<int>& ind0, const std::vector<int>& ind1);
    void sever(const std::vector<int>& ind0, const std::vector<int>& ind1);
    bool isJoined(const std::vector<int>& ind0, const std::vector<int>& ind1) const;
    
    std::vector<double> exec(const std::vector<double>& inputValues) const;
    double deriv(const std::vector<std::vector<int>>& path) const;                // gets chain rule derivative given path of nodes by index
    std::vector<std::vector<int>> derivPath(const std::vector<int>& indNum, const std::vector<int>& indDenom) const; // only functions properly with tree graphs
    
    std::vector<double> gradDescent(
        const std::vector<std::vector<int>>& optimInputInd
    )
    
    /*
    std::vector<double> gradDescent(
        const std::vector<std::vector<int>>& weightInputInd, 
        const std::vector<std::vector<int>>& staticInputInd,
        const std::vector<int>& outputInd,
        const std::vector<std::vector<std::vector<int>>>& weightDerivPath,
        const std::vector<double>& initWeight,
        const std::vector<double>& staticInput,
        const double& alpha,
        const int& maxIteration
    );
    */
};

/* default ctor */
CompGraph::CompGraph()
{
}

/* ctor with shape */
CompGraph::CompGraph(const std::vector<int>& shape)
{
    this->m_numLayers = shape.size();
    this->m_shape = shape;
    for (int i = 0; i < this->m_numLayers; ++i)
    {
        this->m_layers.push_back(std::vector<Node*>(shape[i])); // create vector of nullptrs for each layer
    }
    this->m_numInputs = 0;
    this->m_numOutputs = 0;
}

/* set node */
void CompGraph::set(const std::vector<int>& ind, Node* n)
{
    assert(n != nullptr);
    assert(ind.size() == 2);
    assert((ind[0] >= 0) && (ind[0] < this->m_numLayers));
    assert((ind[1] >= 0) && (ind[1] < this->m_shape[ind[0]]));
    
    this->m_layers[ind[0]][ind[1]] = n;
}

/* set input node */
void CompGraph::setInput(const std::vector<int>& ind, Input* n)
{
    this->set(ind, n);
    this->m_inputs.push_back(n);
    this->m_numInputs++;
}

/* set output node */
void CompGraph::setOutput(const std::vector<int>& ind, Output* n)
{
    this->set(ind, n);
    this->m_outputs.push_back(n);
    this->m_numOutputs++;
}

/* get node */
Node* CompGraph::get(const std::vector<int>& ind) const
{
    assert(ind.size() == 2);
    assert((ind[0] >= 0) && (ind[0] < this->m_numLayers));
    assert((ind[1] >= 0) && (ind[1] < this->m_shape[ind[0]]));
    
    return this->m_layers[ind[0]][ind[1]];
}

/* join node pair (assertion contained in get) 
- node @ ind0 precedes node @ ind1
- indexes in increasing order of layer
*/
void CompGraph::join(const std::vector<int>& ind0, const std::vector<int>& ind1)
{
    assert(ind0.size() == 2);
    assert(ind1.size() == 2);
    assert(ind0[0] < ind1[0]);
    
    this->get(ind1)->addParent(this->get(ind0));
}

/* sever node pair 
- node @ ind0 precedes node @ ind1
- indexes in increasing order of layer
*/
void CompGraph::sever(const std::vector<int>& ind0, const std::vector<int>& ind1)
{
    assert(ind0.size() == 2);
    assert(ind1.size() == 2);
    assert(ind0[0] < ind1[0]);
    
    this->get(ind1)->removeParent(this->get(ind0));
}

/* is joined
- node @ ind0 precedes node @ ind1
- indexes in increasing order of layer
*/
bool CompGraph::isJoined(const std::vector<int>& ind0, const std::vector<int>& ind1) const
{   
    assert(ind0.size() == 2);
    assert(ind1.size() == 2);
    assert(ind0[0] < ind1[0]);
    
    Node* n0 = this->m_layers[ind0[0]][ind0[1]];
    Node* n1 = this->m_layers[ind1[0]][ind1[1]];
    if (n1->findParent(n0) != -1)
    {
        return true;
    }
    return false;
}

/* execute
- input values correspond to each input node in each layer
- returns output values
*/
std::vector<double> CompGraph::exec(const std::vector<double>& inputValues) const
{
    // set inputs
    assert(inputValues.size() == this->m_inputs.size()); 
    for (int i = 0; i < this->m_numInputs; ++i)
    {
        this->m_inputs[i]->set(inputValues[i]);
    }

    // execute each layer
    for (int i = 0; i < this->m_numLayers; ++i)
    {
        for (int j = 0; j < this->m_shape[i]; ++i)
        {
            this->m_layers[i][j]->exec();
        }
    }
    
    // get values of outputs
    std::vector<double> outputValues(this->m_numOutputs);
    for (int i = 0; i < this->m_numOutputs; ++i)
    {
        outputValues[i] = this->m_outputs[i]->value();
    }
    return outputValues;
}

/* derivative 
- uses chain rule
- indexes in increasing order of layer
*/
double CompGraph::deriv(const std::vector<std::vector<int>>& path) const
{
    double d = 1.0; // set to multiplicative identity
    for (int i = 0; i < path.size() - 1; ++i)
    {
        assert(this->isJoined(path[i], path[i + 1]) == true);
        assert(path[i][0] < path[i + 1][0]);
        
        Node* n0 = this->m_layers[path[i][0]][path[i][1]];
        Node* n1 = this->m_layers[path[i + 1][0]][path[i + 1][1]];
        d *= n1->deriv(n0); // get derivative with respect to parent node
    }
    return d;
}

/* derivative path
- gets the path of derivatives for chain rule between a given operation/input (denominator) and operation/output (numerator)
*/
std::vector<std::vector<int>> CompGraph::derivPath(const std::vector<int>& indNum, const std::vector<int>& indDenom) const
{
    std::vector<std::vector<int>> derivPath = {};
    std::vector<std::vector<std::vector<int>>> pathStack;
    
    pathStack.push_back( { indNum } ); // initialise path stack with num node
    while (pathStack.size() != 0)
    {
        // pop the stack 
        std::vector<std::vector<int>> path = pathStack[pathStack.size() - 1];
        path.pop_back();
        
        // get child nodes if this isn't the denom ind
        std::vector<int> pathEndInd = path[path.size() - 1];
        if (pathEndInd == indDenom)
        {
            // break
            derivPath = path;
            break;
        }
        else
        {
            // get 
        }
    }
}


/* gradient descent
- select inputs that are weights
- select inputs that are static
- select output that is optimal at zero - cost function must be scalar (one output)
- assumes tree structure and therefore takes output[0] for derivative chain rule - IMPORTANT
*/
/*
std::vector<double> CompGraph::gradDescent(
    const std::vector<std::vector<int>>& weightInputInd, 
    const std::vector<std::vector<int>>& staticInputInd,
    const std::vector<int>& outputInd,
    const std::vector<std::vector<std::vector<int>>>& weightDerivPath,    
    const std::vector<double>& initWeight,
    const std::vector<double>& staticInput,
    const double& alpha,
    const int& maxIteration
)
{
    assert(weightInputInd.size() + staticInputInd.size() == this->m_layers[0].size()); // ensure consistent with input num
    assert(weightInputInd.size() == initWeight.size());
    assert(staticInputInd.size() == staticInput.size());
    assert(weightDerivPath.size() == weightInputInd.size());
    
    std::vector<double> input(this->m_layers[0].size()); // create empty input vector the size of input layer
    for (int i = 0; i < weightInputInd.size(); ++i)
    {
        assert(weightInputInd[i][0] == 0); // ensure all inputs are in first layer
        input[weightInputInd[i][1]] = initWeight[i]; // set initial weights
    }
    for (int i = 0; i < staticInputInd.size(); ++i)
    {
        assert(staticInputInd[i][0] == 0); // ensure all inputs are in first layer
        input[staticInputInd[i][1]] = staticInput[i]; // set static input
    }
    
    for (int n = 0; n < maxIteration; ++n) // iteration loop
    {
        for (int i = 0; i < weightInputInd.size(); ++i)
        {
            this->get(weightInputInd[i])->exec(input[weightInputInd[i][1]]); // initialise weight inputs with input array
        }
        for (int i = 0; i < staticInputInd.size(); ++i)
        {
            this->get(staticInputInd[i])->exec(input[staticInputInd[i][1]]); // initialise static inputs with input array
        }
    
        std::vector<double> output = this->exec(input); // execute graph with input
        
        std::vector<double> grad(weightInputInd.size()); // create empty vector for gradients for each weight
        for (int i = 0; i < weightInputInd.size(); ++i)
        {
            assert(weightDerivPath[i][0] == weightInputInd[i]);
            assert(weightDerivPath[i][weightDerivPath[i].size() - 1] == outputInd);
            
            grad[i] = this->deriv(weightDerivPath[i]); // get gradient of cost with respect to each weight
        }
        
        for (int i = 0; i < weightInputInd.size(); ++i)
        {
            //this->get(weightInputInd[i])->exec(this->get(weightInputInd[i])->value() + (-1 * alpha * grad[i])); 
            input[weightInputInd[i][1]] += (-1 * alpha * grad[i]); // adjust weight in input array
        }
    }
    
    std::vector<double> finalWeights(weightInputInd.size());
    for (int i = 0; i < weightInputInd.size(); ++i)
    {
        finalWeights[i] = this->get(weightInputInd[i])->value(); // obtain final weights
    }
    
    return finalWeights;
}
*/

