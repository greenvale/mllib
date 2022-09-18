#include <iostream>

#include <matrix.hpp>
#include <linearRegression.hpp>
#include <logisticRegression.hpp>
#include <nn.hpp>
#include <compGraph.hpp>

int main()
{
    CompGraph cg({6, 3, 1, 2, 1, 2});
    
    double weight1 = 0.5;
    double weight2 = 0.5;
    double correct = 1.0;
    
    cg.set({0, 0}, new Input); // input1
    cg.set({0, 1}, new Input); // weighting1
    cg.set({0, 2}, new Input); // input2
    cg.set({0, 3}, new Input); // weighting2
    cg.set({0, 4}, new Input); // correct val
    cg.set({0, 5}, new Input); // -1 error weighting
    
    cg.set({1, 0}, new Mult); // input1 * weighting1
    cg.set({1, 1}, new Mult); // input2 * weighting2
    cg.set({1, 2}, new Mult); // error weighting
    
    cg.set({2, 0}, new Sum); // input1 * weighting1 + input2 * weighting2
    
    cg.set({3, 0}, new Sum); // error calculation 1
    cg.set({3, 1}, new Sum); // error calculation 2
                            
    cg.set({4, 0}, new Mult); // error square
    
    cg.set({5, 0}, new Output); // output
    cg.set({5, 1}, new Output); // error
    
    /* ***************************************************** */
    
    cg.join({{0, 0}, {1, 0}}); // weighting multiply 1
    cg.join({{0, 1}, {1, 0}});
    
    cg.join({{0, 2}, {1, 1}}); // weighting multiply 2
    cg.join({{0, 3}, {1, 1}});
    
    cg.join({{1, 0}, {2, 0}}); // summation
    cg.join({{1, 1}, {2, 0}});
    
    cg.join({{0, 4}, {1, 2}}); // error weighting (-1)
    cg.join({{0, 5}, {1, 2}});
    
    cg.join({{2, 0}, {3, 0}}); // error summation 1
    cg.join({{1, 2}, {3, 0}});
    cg.join({{2, 0}, {3, 1}}); // error summation 2
    cg.join({{1, 2}, {3, 1}});
    
    cg.join({{3, 0}, {4, 0}}); // error square
    cg.join({{3, 1}, {4, 0}});
    
    cg.join({{2, 0}, {5, 0}});
    cg.join({{4, 0}, {5, 1}});
    
    std::vector<double> output = cg.exec({0.0, weight1, 1.0, weight2, correct, -1.0});
    
    std::cout << "Output: " << output[0] << std::endl;
    std::cout << "Error: " << output[1] << std::endl;
    
    /* ***************************************************** */
    
    /*
    double weight1Gradient = cg.deriv({{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 1}});
    double weight2Gradient = cg.deriv({{0, 2}, {1, 1}, {2, 0}, {3, 0}, {4, 0}, {5, 1}});
    std::cout << weight1Gradient << std::endl;
    std::cout << weight2Gradient << std::endl;
    */
    
    std::vector<double> finalWeights = cg.gradDescent(
        {{0, 1}, {0, 3}}, // weight indexes
        {{0, 0}, {0, 2}, {0, 4}, {0, 5}}, // static input indexes
        {5, 1}, // cost output index
        {
            {{0, 1}, {1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 1}}, // deriv path for weight 1
            {{0, 3}, {1, 1}, {2, 0}, {3, 0}, {4, 0}, {5, 1}}  // deriv path for weight 2
        },
        {0.5, 0.5}, // init weights
        {0.0, 1.0, 1.0, -1.0}, // static input value
        0.1,
        1000
    );
    
    for (int i = 0; i < finalWeights.size(); ++i)
    {
        std::cout << finalWeights[i] << std::endl;
    }
    
    std::vector<double> output2 = cg.exec({0.0, finalWeights[0], 1.0, finalWeights[1], correct, -1.0});
    
    std::cout << "Output 2: " << output2[0] << std::endl;
    std::cout << "Error 2: " << output2[1] << std::endl;
    
}

