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
ADD
- exec sums input values into value
- no limit on inputs/outputs
*/ 

template <class T>
class Add : public BackwardNode<T>, public ForwardNode<T>
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

