/*
Computational Graph library
*/

#include <vector>
#include <assert.h>

template <class T>
class Node
{
protected:
    int m_numInputs;
    int m_numOutputs;
    std::vector<Node*> m_inputs;
    std::vector<Node*> m_outputs;
    T m_value; // the value returned by operation
public:
    Node();
    Node(const std::vector<Node*>& inputs, const std::vector<Node*>& outputs);
    
    T value() const;
    int findInput(const Node* n) const;
    int findOutput(const Node* n) const;
    void addInput(const Node* n);
    void addOutput(const Node* n);
    void removeInput(const Node* n);
    void removeOutput(const Node* n);
    void joinInput(Node* n);
    void joinOutput(Node* n);
    void severInput(Node* n);
    void severOutput(Node* n);
    static void join(Node* n1, Node* n2);
    static void sever(Node* n1, Node* n2);
};

// default ctor
template <class T> 
Node<T>::Node()
{
    this->m_inputs = {};
    this->m_outputs = {};
    this->m_value = 0.0;
    this->m_numInputs = 0;
    this->m_numOutputs = 0;
}

// ctor with inputs and outputs
template <class T> 
Node<T>::Node(const std::vector<Node<T>*>& inputs, const std::vector<Node<T>*>& outputs)
{
    this->m_inputs = inputs;
    this->m_outputs = outputs;
    this->m_numInputs = inputs.size();
    this->m_numOutputs = outputs.size();
}

// returns value
template <class T> 
T Node<T>::value() const
{
    return this->m_value;
}

// returns index to node ptr in input array if it exists, else returns -1
template <class T>
int Node<T>::findInput(const Node<T>* n) const
{
    for (int i = 0; i < this->m_numInputs; ++i)
    {
        if (m_inputs[i] == n)
        {
            return i;
        }
    }
    return -1;
}

// returns index to node ptr in output array if it exists, else returns -1
template <class T>
int Node<T>::findOutput(const Node<T>* n) const
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

// add input from node ptr
template <class T>
void Node<T>::addInput(const Node<T>* n)
{
    assert(this->findInput(n) == -1); // make sure input isn't already present
    this->m_inputs.push_back(n);
    this->m_numInputs++;
}

// add output from node ptr
template <class T>
void Node<T>::addOutput(const Node<T>* n)
{
    assert(this->findInput(n) == -1); // make sure input isn't already present
    this->m_outputs.push_back(n);
    this->m_numOutputs++;
}

// remove input from node ptr
template <class T>
void Node<T>::removeInput(const Node<T>* n)
{
    int index = this->findInput(n);
    assert(index != -1); // make sure input is present
    this->m_inputs.erase(this->m_inputs.begin() + index);
    this->m_numInputs--;
}

// remove output from node ptr
template <class T>
void Node<T>::removeOutput(const Node<T>* n)
{
    int index = this->findOutput(n);
    assert(index != -1); // make sure input is present
    this->m_outputs.erase(this->m_outputs.begin() + index);
    this->m_numOutputs--;
}

// join input from node ptr
// assertion contained in addInput/Output
template <class T>
void Node<T>::joinInput(Node<T>* n)
{
    this->addInput(n);
    n->addOutput(this);
}

// join output from node ptr
// assertion contained in addInput/Output
template <class T>
void Node<T>::joinOutput(Node<T>* n)
{
    this->addOutput(n);
    n->addInput(this);
}

// sever input from node ptr
// assertion contained in removeInput/Output
template <class T>
void Node<T>::severInput(Node<T>* n)
{
    this->removeInput(n);
    n->removeOutput(this);
}

// sever output from node ptr
// assertion contained in removeInput/Output
template <class T>
void Node<T>::severOutput(Node<T>* n)
{
    this->removeOutput(n);
    n->removeInput(this);
}

// join a to b where b is a's output and a is b's input
// assertion contained in addInput/Output
template <class T>
void Node<T>::join(Node<T>* n1, Node<T>* n2)
{
    n1->addOutput(n2);
    n2->addInput(n1);
}

// remove connection between a to b
// assertion contained in removeInput/Output
template <class T>
void Node<T>::sever(Node<T>* n1, Node<T>* n2)
{
    n1->removeOutput(n2);
    n2->removeInput(n1);
}

/*
***************************************************************************************************************************
*/ 

template <class T>
class Input : public Node<T>
{
private:
    
public:
    void exec(const T& value);
};

template <class T>
void Input<T>::exec(const T& value)
{
    this->m_value = value;
}

/*
***************************************************************************************************************************
*/ 

template <class T>
class Add : public Node<T>
{
private:
    
public:
    void exec();
};

template <class T>
void Add<T>::exec()
{
    T value = 0.0;
    for (int i = 0; i < this->m_numInputs; ++i)
    {
        value += this->m_inputs[i]->value();
    }
}
