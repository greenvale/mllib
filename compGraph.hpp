/*
Computational Graph library
*/

#include <vector>
#include <assert.h>

/*
***************************************************************************************************************************
NODE
- contains value
*/

template <class T>
class Node
{
protected:
    T m_value;
    int m_numInputs;
    int m_numOutputs;
    std::vector<Node<T>*> m_inputs;
    std::vector<Node<T>*> m_outputs;
public:
    Node();
    
    T value() const;
    int findInput(const Node<T>* n) const;
    int findOutput(const Node<T>* n) const;
    void addInput(Node<T>* n);
    void addOutput(Node<T>* n);
    void removeInput(Node<T>* n);
    void removeOutput(Node<T>* n);
    
    // overriden methods
    virtual void exec(const T& value) {}
    virtual void exec() {}
};

/* ctor */
template <class T>
Node<T>::Node()
{
    this->m_value = 0.0;
    this->m_numInputs = 0;
    this->m_numOutputs = 0;
    this->m_inputs = {};
    this->m_outputs = {};
}

/* returns value of node */
template <class T> 
T Node<T>::value() const
{
    return this->m_value;
}

/* find  input index by node ptr, returns -1 if not found */
template <class T>
int Node<T>::findInput(const Node<T>* n) const
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

/* add input by node ptr */
template <class T>
void Node<T>::addInput(Node<T>* n)
{
    assert(this->findInput(n) == -1); // make sure output isn't already present
    this->m_inputs.push_back(n);
    this->m_numInputs++;
}

/* remove input by node ptr */
template <class T>
void Node<T>::removeInput(Node<T>* n)
{
    int index = this->findInput(n);
    assert(index != -1); // make sure input is present
    this->m_inputs.erase(this->m_inputs.begin() + index);
    this->m_numInputs--;
}

/* find output index by node ptr, returns -1 if not found */
template <class T>
int Node<T>::findOutput(const Node<T>* n) const
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

/* add output by node ptr */
template <class T>
void Node<T>::addOutput(Node<T>* n)
{
    assert(this->findOutput(n) == -1); // make sure output isn't already present
    this->m_outputs.push_back(n);
    this->m_numOutputs++;
}

/* remove output by node ptr */
template <class T>
void Node<T>::removeOutput(Node<T>* n)
{
    int index = this->findOutput(n);
    assert(index != -1); // make sure input is present
    this->m_outputs.erase(this->m_outputs.begin() + index);
    this->m_numOutputs--;
}

/*
***************************************************************************************************************************
INPUT
- 
*/

template <class T>
class Input : public Node<T>
{
private:
public:
    void exec(const T& value);
};

/* execute */
template <class T>
void Input<T>::exec(const T& value)
{
    assert(this->m_numInputs == 0); // check node configuration is correct
    this->m_value = value;
}

/*
***************************************************************************************************************************
OUTPUT
- 
*/

template <class T>
class Output : public Node<T>
{
private:
public:
    void exec();
};

/* execute */
template <class T>
void Output<T>::exec()
{
    assert(this->m_numInputs == 1); // check node configuration is correct
    assert(this->m_numOutputs == 0);
    this->m_value = this->m_inputs[0]->value();
}

/*
***************************************************************************************************************************
ADD
- exec sums input values into value
*/ 

template <class T>
class Sum : public Node<T>
{
private:
public:
    void exec();
};

/* execute */
template <class T>
void Sum<T>::exec()
{
    this->m_value = 0.0; // set to identity
    for (int i = 0; i < this->m_numInputs; ++i)
    {
        this->m_value += this->m_inputs[i]->value();
    }
}

/*
***************************************************************************************************************************
MULTIPLY
- exec multiplies input values into value
*/ 

template <class T>
class Mult : public Node<T>
{
private:
public:
    void exec();
};

/* execute */
template <class T>
void Mult<T>::exec()
{
    this->m_value = 1.0; // set to identity
    for (int i = 0; i < this->m_numInputs; ++i)
    {
        this->m_value *= this->m_inputs[i]->value();
    }
}

/*
***************************************************************************************************************************
COMPUTATIONAL GRAPH
- the first layer is inputs only, the last layer is outputs only
- middle layers contain operations
- each layer contains nodes that are independent of eachother meaning they can be executed in any order
- there is currently no enforcement of input-operation-output structure (however, it will exec if not done properly)
*/ 

template <class T>
class CompGraph
{
private:
    int m_numLayers;
    std::vector<int> m_shape;
    std::vector<std::vector<Node<T>*>> m_layers;
public:
    CompGraph();
    CompGraph(const std::vector<int>& shape);
    
    void set(const std::vector<int>& ind, Node<T>* n);
    Node<T>* get(const std::vector<int>& ind);
    void join(const std::vector<int>& ind0, const std::vector<int>& ind1);
    void sever(const std::vector<int>& ind0, const std::vector<int>& ind1);
    
    std::vector<T> exec(const std::vector<T>& input);
};

/* default ctor */
template <class T>
CompGraph<T>::CompGraph()
{
}

/* ctor with shape */
template <class T>
CompGraph<T>::CompGraph(const std::vector<int>& shape)
{
    m_numLayers = shape.size();
    m_shape = shape;
    for (int i = 0; i < m_numLayers; ++i)
    {
        std::vector<Node<T>*> vec(shape[i]);
        m_layers.push_back(vec); // create vector of nullptrs for each layer
    }
}

/* set node */
template <class T>
void CompGraph<T>::set(const std::vector<int>& ind, Node<T>* n)
{
    assert(n != nullptr);
    assert(ind.size() == 2);
    assert((ind[0] >= 0) && (ind[0] < this->m_numLayers));
    assert((ind[1] >= 0) && (ind[1] < this->m_shape[ind[0]]));
    
    this->m_layers[ind[0]][ind[1]] = n;
}

/* get node */
template <class T>
Node<T>* CompGraph<T>::get(const std::vector<int>& ind)
{
    assert(ind.size() == 2);
    assert((ind[0] >= 0) && (ind[0] < this->m_numLayers));
    assert((ind[1] >= 0) && (ind[1] < this->m_shape[ind[0]]));
    
    return this->m_layers[ind[0]][ind[1]];
}

/* join node pair (assertion contained in get) 
- node @ ind0 precedes node @ ind1
*/
template <class T>
void CompGraph<T>::join(const std::vector<int>& ind0, const std::vector<int>& ind1)
{
    this->get(ind0)->addOutput(this->get(ind1));
    this->get(ind1)->addInput(this->get(ind0));
}

/* sever node pair 
- node @ ind0 precedes node @ ind1
*/
template <class T>
void CompGraph<T>::sever(const std::vector<int>& ind0, const std::vector<int>& ind1)
{
    this->get(ind0)->removeOutput(this->get(ind1));
    this->get(ind1)->removeInput(this->get(ind0));
}

/* execute */
template <class T>
std::vector<T> CompGraph<T>::exec(const std::vector<T>& input) 
{
    assert(input.size() == m_shape[0]);
    
    std::vector<T> output(m_shape[m_numLayers - 1]);
    
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

