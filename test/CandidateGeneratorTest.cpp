#include <iostream>
#include <gtest/gtest.h>

#include "../src/Models/Tools/CandidateGenerator.hpp"


TEST(CandidateGenerator, testCandidateGenerator){
    // use clio input output
    CandidateGenerator cg(3,{{0,2},{0,2}, {0,2}});
    
    std::vector<std::vector<int>> t = {{0,0,0},{1,0,2},{2,2,2}};
    cg.addQueriedPoint(t[0]);
    cg.addQueriedPoint(t[1]);
    cg.addQueriedPoint(t[2]);

    auto cands = cg.getCandidates(9);

    for (auto cand : cands){
        ASSERT_EQ(false, (cand == t[0] || cand == t[1] || cand == t[2]));
    }
}