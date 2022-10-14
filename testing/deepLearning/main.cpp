#include <iostream>

/*
#include <matrix.hpp>
#include <linearRegression.hpp>
#include <logisticRegression.hpp>
#include <nn.hpp>
*/
#include <compGraph2.hpp>



int main()
{
    CompGraph cg({6, 3, 1, 1, 1});
    
    Constant constantOperator;
    Mult multiplyOperator;
    Sum sumOperator;
    Power squareOperator(2.0);
    
    cg.set({0, 0}, &constantOperator, INPUT); // w0
    cg.set({0, 1}, &constantOperator, INPUT); // w1
    cg.set({0, 2}, &constantOperator, INPUT); // x0
    cg.set({0, 3}, &constantOperator, INPUT); // x1
    cg.set({0, 4}, &constantOperator, INPUT); // y
    cg.set({0, 5}, &constantOperator, INPUT); // -1
    
    cg.set({1, 0}, &multiplyOperator, STEP); 
    cg.set({1, 1}, &multiplyOperator, STEP);
    cg.set({1, 2}, &multiplyOperator, STEP);
    
    cg.set({2, 0}, &sumOperator, OUTPUT);
    cg.set({3, 0}, &sumOperator, STEP);
    cg.set({4, 0}, &squareOperator, OUTPUT);
    
    cg.join({{0, 0}, {0, 2}}, {{1, 0}});
    cg.join({{0, 1}, {0, 3}}, {{1, 1}});
    cg.join({{0, 4}, {0, 5}}, {{1, 2}});
    cg.join({{1, 0}, {1, 1}}, {{2, 0}});
    cg.join({{2, 0}, {1, 2}}, {{3, 0}});
    cg.join({{3, 0}}, {{4, 0}});
    
    /*
    std::vector<double> optimInput = cg.gradDescent(
        {{0, 0}, {0, 1}},
        {{0, 2}, {0, 3}, {0, 4}, {0, 5}},
        {4, 0},
        0.1,
        0.001,
        100,
        {0.0, 1.0},
        {{0.0, 1.0, 1.0, -1.0}}
    );
    */
}

/* ********************************************************************************************** */

