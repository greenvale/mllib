
#include <vector>
#include <iostream>
#include "../../compGraph.hpp"

int main()
{
    // operations
    Sum* sum = new Sum();
    Mul* mul = new Mul();
    Squ* squ = new Squ();
    
    std::vector<AdjListElem*> adjList;

    adjList.push_back(new AdjListElem(Pos(0, 0), {}, {Pos(1, 0)}, nullptr));
    adjList.push_back(new AdjListElem(Pos(0, 1), {}, {Pos(1, 0)}, nullptr));
    adjList.push_back(new AdjListElem(Pos(1, 0), {Pos(0, 0), Pos(0, 1)}, {Pos(2, 0)}, mul));
    adjList.push_back(new AdjListElem(Pos(2, 0), {Pos(1, 0)}, {}, squ));

    CompGraph cg({2, 1, 1}, adjList);
    cg.reset();
    cg.writeVal(Pos(0, 0), 2.0);
    cg.writeVal(Pos(0, 1), 3.0);
    cg.exec();
    std::cout << cg.readVal(Pos(2, 0)) << std::endl;
    std::cout << cg.readDeriv(Pos(2, 0), 0) << std::endl;

    /*
    DerivChain dc = cg.getChain(Pos(0, 1), Pos(2, 0));
    for (int i = 0; i < dc.m_nodeArr.size(); ++i)
    {
        std::cout << "Node pos: " << dc.m_nodeArr[i]->m_pos.m_col << ", " << dc.m_nodeArr[i]->m_pos.m_row << " @ index " << dc.m_indArr[i] << std::endl;
    }
    std::cout << cg.chainDeriv(dc) << std::endl;
    */

    cg.optimise(
        {Pos(0, 0)},
        {Pos(0, 1)},
        Pos(2, 0),
        {0.5},
        {{{2.0}}}
    );

    std::cout << cg.readVal(Pos(2, 0)) << std::endl;
    std::cout << cg.readVal(Pos(0, 0)) << std::endl;
}