/*
Computational Graph library
- double precision
- nodes are doubly linked
*/

#include <vector>
#include <assert.h>

/*
***************************************************************************************************************************
NODE
- contains value
*/

class Node
{
protected:
    double m_value;
    int m_numInputs;
    int m_numOutputs;
    std::vector<Node*> m_inputs;
    std::vector<Node*> m_outputs;
public:
    Node();
    
    double value() const;
    int findInput(const Node* n) const;
    int findOutput(const Node* n) const;
    void addInput(Node* n);
    void addOutput(Node* n);
    void removeInput(Node* n);
    void removeOutput(Node* n);
    
    // overriden methods
    virtual void exec(const double& value) {}
    virtual void exec() {}
    virtual double deriv(const Node* n);
};

/* ctor */
Node::Node()
{
    this->m_value = 0.0;
    this->m_numInputs = 0;
    this->m_numOutputs = 0;
    this->m_inputs = {};
    this->m_outputs = {};
}

/* returns value of node */
double Node::value() const
{
    return this->m_value;
}

/* find  input index by node ptr, returns -1 if not found */
int Node::findInput(const Node* n) const
{
    assert(n != nullptr);
    for (int i = 0; i < this->m_numInputs; ++i)
    {
        if (this->m_inputs[i] == n)
        {
            return i;
        }
    }
    return -1;
}

/* find output index by node ptr, returns -1 if not found */
int Node::findOutput(const Node* n) const
{
    assert(n != nullptr);
    for (int i = 0; i < this->m_numOutputs; ++i)
    {
        if (this->m_outputs[i] == n)
        {
            return i;
        }
    }
    return -1;
}

/* add input by node ptr */
void Node::addInput(Node* n)
{
    assert(n != this);
    assert(this->findInput(n) == -1); // make sure output isn't already present
    this->m_inputs.push_back(n);
    this->m_numInputs++;
}

/* add output by node ptr */
void Node::addOutput(Node* n)
{
    assert(n != this);
    assert(this->findOutput(n) == -1); // make sure output isn't already present
    this->m_outputs.push_back(n);
    this->m_numOutputs++;
}

/* remove input by node ptr */
void Node::removeInput(Node* n)
{
    int index = this->findInput(n);
    assert(index != -1); // make sure input is present
    this->m_inputs.erase(this->m_inputs.begin() + index);
    this->m_numInputs--;
}

/* remove output by node ptr */
void Node::removeOutput(Node* n)
{
    int index = this->findOutput(n);
    assert(index != -1); // make sure input is present
    this->m_outputs.erase(this->m_outputs.begin() + index);
    this->m_numOutputs--;
}

/* derivative */
double Node::deriv(const Node* n)
{
    return 0.0;
}

/*
***************************************************************************************************************************
INPUT
- 
*/

class Input : public Node
{
private:
public:
    void exec(const double& value);
};

/* execute */
void Input::exec(const double& value)
{
    assert(this->m_numInputs == 0); // check node configuration is correct
    this->m_value = value;
}

/*
***************************************************************************************************************************
OUTPUT
- 
*/

class Output : public Node
{
private:
public:
    void exec();
    double deriv(const Node* n);
};

/* execute */
void Output::exec()
{
    assert(this->m_numInputs == 1); // check node configuration is correct
    assert(this->m_numOutputs == 0);
    this->m_value = this->m_inputs[0]->value();
}

/* derivative 
- always constant 1
*/
double Output::deriv(const Node* n)
{
    return 1.0;
}

/*
***************************************************************************************************************************
ADD
- exec sums input values into value
*/ 

class Sum : public Node
{
private:
public:
    void exec();
    double deriv(const Node* n);
};

/* execute */
void Sum::exec()
{
    this->m_value = 0.0; // set to additive identity
    for (int i = 0; i < this->m_numInputs; ++i)
    {
        this->m_value += this->m_inputs[i]->value();
    }
}

/* derivative 
- always constant 1
*/
double Sum::deriv(const Node* n)
{
    return 1.0;
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
    double deriv(const Node* n);
};

/* execute */
void Mult::exec()
{
    this->m_value = 1.0; // set to multiplicative identity
    for (int i = 0; i < this->m_numInputs; ++i)
    {
        this->m_value *= this->m_inputs[i]->value();
    }
}

/* derivative 
- differentiates operator with respect to input (multiple of other inputs values)
*/
double Mult::deriv(const Node* n)
{
    double d = 1.0; // set to multiplicative identity
    for (int i = 0; i < this->m_numInputs; ++i)
    {
        if (this->m_inputs[i] != n)
        {
            d *= this->m_inputs[i]->value();
        }
    }
    return d;
}

/*
***************************************************************************************************************************
COMPUTATIONAL GRAPH
- the first layer is inputs only, the last layer is outputs only
- middle layers contain operations
- each layer contains nodes that are independent of eachother meaning they can be executed in any order
- there is currently no enforcement of input-operation-output structure (however, it will exec if not done properly)
*/ 

class CompGraph
{
private:
    int m_numLayers;
    std::vector<int> m_shape;
    std::vector<std::vector<Node*>> m_layers; // first layer is input layer, last layer is output layer
public:
    CompGraph();
    CompGraph(const std::vector<int>& shape);
    
    void set(const std::vector<int>& ind, Node* n);
    Node* get(const std::vector<int>& ind);
    void join(const std::vector<std::vector<int>>& ind);
    void sever(const std::vector<std::vector<int>>& ind);
    bool isJoined(const std::vector<std::vector<int>>& ind);
    
    std::vector<double> exec(const std::vector<double>& input);
    double deriv(const std::vector<std::vector<int>>& ind);
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

/* get node */
Node* CompGraph::get(const std::vector<int>& ind)
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
void CompGraph::join(const std::vector<std::vector<int>>& ind)
{
    assert(ind.size() == 2);
    assert(ind[0][0] < ind[1][0]);
    
    this->get(ind[0])->addOutput(this->get(ind[1]));
    this->get(ind[1])->addInput(this->get(ind[0]));
}

/* sever node pair 
- node @ ind0 precedes node @ ind1
- indexes in increasing order of layer
*/
void CompGraph::sever(const std::vector<std::vector<int>>& ind)
{
    assert(ind.size() == 2);
    assert(ind[0][0] < ind[1][0]);
    
    this->get(ind[0])->removeOutput(this->get(ind[1]));
    this->get(ind[1])->removeInput(this->get(ind[0]));
}

/* is joined
- node @ ind0 precedes node @ ind1
- indexes in increasing order of layer
*/
bool CompGraph::isJoined(const std::vector<std::vector<int>>& ind)
{   
    assert(ind.size() == 2);
    assert(ind[0][0] < ind[1][0]);
    
    Node* n0 = this->m_layers[ind[0][0]][ind[0][1]];
    Node* n1 = this->m_layers[ind[1][0]][ind[1][1]];
    if ((n0->findOutput(n1) != -1) && (n1->findInput(n0) != -1))
    {
        return true;
    }
    return false;
}

/* execute */
std::vector<double> CompGraph::exec(const std::vector<double>& input) 
{
    assert(input.size() == this->m_shape[0]);
    
    std::vector<double> output(this->m_shape[m_numLayers - 1]);
    for (int i = 0; i < this->m_numLayers; ++i)
    {
        for (int j = 0; j < this->m_shape[i]; ++j)
        {
            if (i == 0) // input layer
            {
                this->get({i, j})->exec(input[j]);
            }
            else if (i == m_numLayers - 1) // output layer
            {
                this->get({i, j})->exec();
                output[j] = this->get({i, j})->value();
            }
            else // operation layer
            {
                this->get({i, j})->exec();
            }
        }
    }
    return output;
}

/* derivative 
- uses chain rule
- indexes in increasing order of layer
*/
double CompGraph::deriv(const std::vector<std::vector<int>>& path)
{
    double d = 1.0; // set to multiplicative identity
    for (int i = 0; i < path.size() - 1; ++i)
    {
        assert(this->isJoined({path[i], path[i + 1]}) == true);
        assert(path[i][0] < path[i + 1][0]);
        
        Node* n0 = this->m_layers[path[i][0]][path[i][1]];
        Node* n1 = this->m_layers[path[i + 1][0]][path[i + 1][1]];
        d *= n1->deriv(n0);
    }
    return d;
}

/* gradient descent
- select inputs that are weights
- select inputs that are static
- select output that is optimal at zero - cost function must be scalar (one output)
- assumes tree structure and therefore takes output[0] for derivative chain rule - IMPORTANT
*/
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

