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
public:
    virtual T value() const;
};

/* returns value of node */
template <class T> 
T Node<T>::value() const
{
    return this->m_value;
}

/*
***************************************************************************************************************************
BACKWARD NODE
- contains inputs
*/

template <class T>
class BackwardNode : virtual public Node<T>
{
protected:
    int m_numInputs;
    std::vector<Node<T>*> m_inputs;
public:
    virtual int findInput(const Node<T>* n) const;
    virtual void addInput(Node<T>* n);
    virtual void removeInput(Node<T>* n);
};

/* find  input index by node ptr, returns -1 if not found */
template <class T>
int BackwardNode<T>::findInput(const Node<T>* n) const
{
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
void BackwardNode<T>::addInput(Node<T>* n)
{
    assert(this->findInput(n) == -1); // make sure output isn't already present
    this->m_inputs.push_back(n);
    this->m_numInputs++;
}

/* remove input by node ptr */
template <class T>
void BackwardNode<T>::removeInput(Node<T>* n)
{
    int index = this->findInput(n);
    assert(index != -1); // make sure input is present
    this->m_inputs.erase(this->m_inputs.begin() + index);
    this->m_numInputs--;
}

/*
***************************************************************************************************************************
FORWARD NODE
- contains outputs
*/

template <class T>
class ForwardNode : virtual public Node<T>
{
protected:
    int m_numOutputs;
    std::vector<Node<T>*> m_outputs;
public:
    virtual int findOutput(const Node<T>* n) const;
    virtual void addOutput(Node<T>* n);
    virtual void removeOutput(Node<T>* n);
};

/* find output index by node ptr, returns -1 if not found */
template <class T>
int ForwardNode<T>::findOutput(const Node<T>* n) const
{
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
void ForwardNode<T>::addOutput(Node<T>* n)
{
    assert(this->findOutput(n) == -1); // make sure output isn't already present
    this->m_outputs.push_back(n);
    this->m_numOutputs++;
}

/* remove output by node ptr */
template <class T>
void ForwardNode<T>::removeOutput(Node<T>* n)
{
    int index = this->findOutput(n);
    assert(index != -1); // make sure input is present
    this->m_outputs.erase(this->m_outputs.begin() + index);
    this->m_numOutputs--;
}

/*
***************************************************************************************************************************
INPUT
- exec writes to value from argument
- no inputs allowed
- multiply outputs allowed
*/ 

template <class T>
class Input : public ForwardNode<T>
{
private:
public:
    Input();
    void exec(const T& value);
};

/* ctor */
template <class T>
Input<T>::Input()
{
    this->m_numOutputs = 0;
    this->m_outputs = {};
    this->m_value = 0.0;
}

/* execute */
template <class T>
void Input<T>::exec(const T& value)
{
    this->m_value = value;
}

/*
***************************************************************************************************************************
OUTPUT
- exec writes to value from single input value
- only one input allowed
- no outputs allowed
*/ 

template <class T>
class Output : public BackwardNode<T>
{
private:
public:
    Output();
    void addInput(Node<T>* n); // override add input to only allow 1 input
    void exec();
};

/* ctor */
template <class T>
Output<T>::Output()
{
    this->m_numInputs = 0;
    this->m_inputs = {};
    this->m_value = 0.0;
}

/* execute */
template <class T>
void Output<T>::exec()
{
    this->m_value = this->m_inputs[0]->value();
}

/* add input override (only 1 input allowed) */
template <class T>
void Output<T>::addInput(Node<T>* n)
{
    assert(this->findInput(n) == -1); // make sure output isn't already present
    assert(this->m_numInputs == 0); // only one input is allowed
    
    this->m_inputs.push_back(n);
    this->m_numInputs++;
}

/*
***************************************************************************************************************************
OPERATION
- no limit on inputs/outputs
*/ 

template <class T>
class Operation : public BackwardNode<T>, public ForwardNode<T>
{
private:
public:
    virtual void exec();
};

template <class T>
void Operation<T>::exec() 
{
}

/*
***************************************************************************************************************************
ADD
- exec sums input values into value
*/ 

template <class T>
class Add : public Operation<T>
{
private:
public:
    Add();
    void exec();
};

/* ctor */
template <class T>
Add<T>::Add()
{
    this->m_numInputs = 0;
    this->m_numOutputs = 0;
    this->m_inputs = {};
    this->m_outputs = {};
    this->m_value = 0.0;
}

/* execute */
template <class T>
void Add<T>::exec()
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
class Mult : public Operation<T>
{
private:
public:
    Mult();
    void exec();
};

/* ctor */
template <class T>
Mult<T>::Mult()
{
    this->m_numInputs = 0;
    this->m_numOutputs = 0;
    this->m_inputs = {};
    this->m_outputs = {};
    this->m_value = 0.0;
}

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
- structure of nodes, doesn't contain nodes - only ptrs
- nodes must either be declared on stack or heap seperately
*/ 

template <class T>
class CompGraph
{
private:
    int m_numInputs;
    int m_numOperations;
    int m_numOutputs;
    std::vector<Input<T>*> m_inputs;
    std::vector<Output<T>*> m_outputs;
    std::vector<Operation<T>*> m_operations;
public:
    CompGraph();
    
    // find/add/remove input/operation/output
    int findInput(Input<T>* n);
    int findOutput(Output<T>* n);
    int findOperation(Operation<T>* n);
    void addInput(Input<T>* n);
    void addOutput(Output<T>* n);
    void addOperation(Operation<T>* n);
    void removeInput(Input<T>* n);
    void removeOutput(Output<T>* n);
    void removeOperation(Operation<T>* n);
    
    void join(Input<T>* n0, Operation<T>* n1);
    void join(Input<T>* n0, Output<T>* n1);
    void join(Operation<T>* n0, Operation<T>* n1);
    void join(Operation<T>* n0, Output<T>* n1);
    
    void sever(Input<T>* n0, Operation<T>* n1);
    void sever(Input<T>* n0, Output<T>* n1);
    void sever(Operation<T>* n0, Operation<T>* n1);
    void sever(Operation<T>* n0, Output<T>* n1);
    
    void exec(const std::vector<T>& v);
};

/* ctor */
template <class T>
CompGraph<T>::CompGraph()
{
    m_numInputs = 0;
    m_numOutputs = 0;
    m_numOperations = 0;
    m_inputs = {};
    m_outputs = {};
    m_operations = {};
}

/* ************************************************************************ */

/* find input by input node ptr */
template <class T>
int CompGraph<T>::findInput(Input<T>* n)
{
    for (int i = 0; i < this->m_numInputs; ++i)
    {
        if (this->m_inputs[i] == n)
        {
            return i;
        }
    }
    return -1;
}

/* find output by output node ptr */
template <class T>
int CompGraph<T>::findOutput(Output<T>* n)
{
    for (int i = 0; i < this->m_numOutputs; ++i)
    {
        if (this->m_outputs[i] == n)
        {
            return i;
        }
    }
    return -1;
}

/* find operation by operation node ptr */
template <class T>
int CompGraph<T>::findOperation(Operation<T>* n)
{
    for (int i = 0; i < this->m_numOperations; ++i)
    {
        if (this->m_operations[i] == n)
        {
            return i;
        }
    }
    return -1;
}

/* add input by input node ptr */
template <class T>
void CompGraph<T>::addInput(Input<T>* n)
{
    assert(this->findInput(n) == -1);
    this->m_inputs.push_back(n);
    this->m_numInputs++;
}

/* add output by output node ptr */
template <class T>
void CompGraph<T>::addOutput(Output<T>* n)
{
    assert(this->findOutput(n) == -1);
    this->m_outputs.push_back(n);
    this->m_numOutputs++;
}

/* add operation by operation node ptr */
template <class T>
void CompGraph<T>::addOperation(Operation<T>* n)
{
    assert(this->findOperation(n) == -1);
    this->m_operations.push_back(n);
    this->m_numOperations++;
}

/* remove input by input node ptr */
template <class T>
void CompGraph<T>::removeInput(Input<T>* n)
{
    int index = this->findInput(n);
    assert(index != -1); // make sure input is present
    this->m_inputs.erase(this->m_inputs.begin() + index);
    this->m_numInputs--;
}

/* remove output by output node ptr */
template <class T>
void CompGraph<T>::removeOutput(Output<T>* n)
{
    int index = this->findOutput(n);
    assert(index != -1); // make sure input is present
    this->m_outputs.erase(this->m_outputs.begin() + index);
    this->m_numOutputs--;
}

/* remove operation by operation node ptr */
template <class T>
void CompGraph<T>::removeOperation(Operation<T>* n)
{
    int index = this->findOperation(n);
    assert(index != -1); // make sure input is present
    this->m_operations.erase(this->m_operations.begin() + index);
    this->m_numOperations--;
}

/* ************************************************************************ */
/* Node - Node functions */

/* join input node to operation node */
template <class T>
void CompGraph<T>::join(Input<T>* n0, Operation<T>* n1)
{
    assert(this->findInput(n0) != -1); // check both nodes are in the graph
    assert(this->findOperation(n1) != -1);
    n0->addOutput(n1);
    n1->addInput(n0);
}

/* sever */
template <class T>
void CompGraph<T>::sever(Input<T>* n0, Operation<T>* n1)
{
    assert(this->findInput(n0) != -1); // check both nodes are in the graph
    assert(this->findOperation(n1) != -1);
    n0->removeOutput(n1);
    n1->removeInput(n0);
}

/* join input node to output node */
template <class T>
void CompGraph<T>::join(Input<T>* n0, Output<T>* n1)
{
    assert(this->findInput(n0) != -1); // check both nodes are in the graph
    assert(this->findOutput(n1) != -1);
    n0->addOutput(n1);
    n1->addInput(n0);
}

/* sever input node from output node */

/* join operation node to operation node */
template <class T>
void CompGraph<T>::join(Operation<T>* n0, Operation<T>* n1)
{
    assert(this->findOperation(n0) != -1); // check both nodes are in the graph
    assert(this->findOperation(n1) != -1);
    n0->addOutput(n1);
    n1->addInput(n0);
}

/* sever operation node from operation node */ 

/* join operation node to output node */
template <class T>
void CompGraph<T>::join(Operation<T>* n0, Output<T>* n1)
{
    assert(this->findOperation(n0) != -1); // check both nodes are in the graph
    assert(this->findOutput(n1) != -1);
    n0->addOutput(n1);
    n1->addInput(n0);
}

/* sever operation node from output node */


/* ************************************************************************ */

/* execute */
template <class T>
void CompGraph<T>::exec(const std::vector<T>& v)
{
    assert(v.size() == this->m_numInputs);
    
    // inputs
    for (int i = 0; i < m_numInputs; ++i)
    {
        m_inputs[i]->exec(v[i]);
    }
    
    // operations
    std::vector<int> execFlags(this->m_numOperations, 0); // flags to show if operation has executed
    int complete = 0;
    std::vector<Operation<T>*> operationStack = {}; // create empty vector for operation stack
    operationStack.push_back(m_operations[0]);
    
    while (complete == 0)
    {
        // take top element in stack
        int indexTop = operationStack.size() - 1;   
        
        // check which inputs to this operation are not executed and add them to stack
        int ready = 1; 
        for (int i = 0; i < operationStack[indexTop]->numInputs(); ++i)
        {
            if ()
            {
                
            }
        }
        
        // if all inputs are executed then execute, else quit
    
        // check if complete
        complete = 1;
    }
    
    // outputs
}
