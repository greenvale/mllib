#include <vector>
#include <iostream>
#include "../../NeuralNetwork.hpp"
#include "../../LogisticRegression.hpp"
#include "../../../mathlib/LinearAlgebra.hpp"
#include "math.h"

int main()
{


    //NeuralNetwork nn({3, 4, 2, 2});
    NeuralNetwork nn({2, 3, 1});

    //nn.train(mathlib::Matrix({3,1}, {{2.0},{1.0},{1.0}}), mathlib::Matrix({2,1}, {{1.0},{0.0}}), 0.1, 0.001, 1000);

    nn.train({{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}}, {{0.0}, {1.0}, {1.0}, {1.0}}, 0.1, 0.001, 10000);

    (nn.evaluate(mathlib::Matrix({2,1},{{0.0},{0.0}}))).display();
    (nn.evaluate(mathlib::Matrix({2,1},{{0.0},{1.0}}))).display();
    (nn.evaluate(mathlib::Matrix({2,1},{{1.0},{0.0}}))).display();
    (nn.evaluate(mathlib::Matrix({2,1},{{1.0},{1.0}}))).display();

    //nn.display();


/*
    LogisticReg lr(3, 2);

    lr.train(mathlib::Matrix({3,1}, {{2.0},{1.0},{1.0}}), mathlib::Matrix({2,1}, {{1.0},{0.0}}), 0.1, 0.001, 10000);
*/

}