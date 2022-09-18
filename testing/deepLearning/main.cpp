#include <iostream>

#include <matrix.hpp>
#include <linearRegression.hpp>
#include <logisticRegression.hpp>
#include <nn.hpp>
#include <compGraph.hpp>

int main()
{
    CompGraph cg({6, 3, 1, 1, 2});
    
    double weight1 = 0.5;
    double weight2 = 0.5;
    double correct = 1.0;
    
    cg.set({0, 0}, new Input);
    cg.set({0, 1}, new Input); // weighting
    cg.set({0, 2}, new Input);
    cg.set({0, 3}, new Input); // weighting
    cg.set({0, 4}, new Input); // correct
    cg.set({0, 5}, new Input); // -1 error weighting
    
    cg.set({1, 0}, new Mult);
    cg.set({1, 1}, new Mult);
    cg.set({1, 2}, new Mult); // error weighting
    
    cg.set({2, 0}, new Sum);
    
    cg.set({3, 0}, new Sum);
    
    cg.set({4, 0}, new Output); // output
    cg.set({4, 1}, new Output); // error
    
    cg.join({{0, 0}, {1, 0}}); // weighting multiply 1
    cg.join({{0, 1}, {1, 0}});
    
    cg.join({{0, 2}, {1, 1}}); // weighting multiply 2
    cg.join({{0, 3}, {1, 1}});
    
    cg.join({{1, 0}, {2, 0}}); // summation
    cg.join({{1, 1}, {2, 0}});
    
    cg.join({{0, 4}, {1, 2}}); // error weighting (-1)
    cg.join({{0, 5}, {1, 2}});
    
    cg.join({{2, 0}, {3, 0}}); // error summation
    cg.join({{1, 2}, {3, 0}});
    
    cg.join({{2, 0}, {4, 0}});
    cg.join({{3, 0}, {4, 1}});
    
    std::vector<double> output = cg.exec({0.0, weight1, 1.0, weight2, correct, -1.0});
    
    std::cout << "Output: " << output[0] << std::endl;
    std::cout << "Error: " << output[1] << std::endl;
    
}
