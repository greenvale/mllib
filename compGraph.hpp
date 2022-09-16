/*
Computational Graph library
*/

#include <vector>
#include <assert.h>

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
*/

template <class T>
class ForwardNode : public Node<T>
{
protected:
    int m_numOutputs;
    std::vector<Node<T>*> m_outputs;
public:
    virtual int findOutput(const Node<T>* n) const;
    virtual void addOutput(Node<T>* n);
    virtual void removeOutput(Node<T>* n);
};

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

template <class T>
void ForwardNode<T>::addOutput(Node<T>* n)
{
    assert(this->findOutput(n) == -1); // make sure output isn't already present
    this->m_outputs.push_back(n);
    this->m_numOutputs++;
}

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
*/ 

template <class T>
class Input : public ForwardNode<T>
{
private:
public:
    Input();
    void exec(const T& value);
};

template <class T>
Input<T>::Input()
{
    this->m_numOutputs = 0;
    this->m_outputs = {};
    this->m_value = 0.0;
}

template <class T>
void Input<T>::exec(const T& value)
{
    this->m_value = value;
}

