#include <iostream>
#include <gtest/gtest.h>

#include "../src/InputOutput/InputOutput.hpp"
#include "../src/InputOutput/CommandLineInputOutput.hpp"

using namespace std;

TEST(TestCommandLine, TestsModel1D){
    CommandLineInputOutput myModel;

    // test functionality
    EXPECT_EQ(1, myModel.get_next_query().size());
    EXPECT_EQ(0, myModel.get_state_space().get({1}));
    EXPECT_NO_THROW(myModel.update_prediction({1}, 1));
    EXPECT_EQ(1, myModel.get_state_space().get({1}));

    // check that created array is correct size
    EXPECT_NO_THROW(myModel.get_state_space().get({0}));
    EXPECT_NO_THROW(myModel.get_state_space().get({9}));
    EXPECT_THROW(myModel.get_state_space().get({10}), std::out_of_range);
}

TEST(TestModel, TestsModelND){
    for (int i = 1; i < 6; i++){
        DumbModel myModel(i, 10);
        
        vector<int> queryPoint(i, 0);

        // test functionality
        EXPECT_EQ(i, myModel.get_next_query().size());
        EXPECT_EQ(0, myModel.get_state_space().get(queryPoint));
        EXPECT_NO_THROW(myModel.update_prediction(queryPoint, 1));
        EXPECT_EQ(1, myModel.get_state_space().get(queryPoint));

        // check that created array is correct size
        for (int j = 0; j < i; j++){
            EXPECT_NO_THROW(myModel.get_state_space().get(queryPoint));
            queryPoint[j] = 9;
            EXPECT_NO_THROW(myModel.get_state_space().get(queryPoint));
            queryPoint[j] = 10;
            EXPECT_THROW(myModel.get_state_space().get(queryPoint), std::out_of_range);
            queryPoint[j] = 9;
        }
    }
}