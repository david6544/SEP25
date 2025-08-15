#include <iostream>
#include <gtest/gtest.h>

#include "../StateSpace/StateSpace.hpp"

using namespace std;

TEST(OneDimensionAccess, TestsGetAndSetIn1D){
    StateSpace mySpace(1, 5);
    ASSERT_EQ(0.0, mySpace.get({1}));
    mySpace.set({1}, -1);
    ASSERT_EQ(-1.0, mySpace.get({1}));
}

TEST(MultDimensionAccess, TestsGetAndSetInND){
    for (int i = 1; i < 6; i++){
        StateSpace mySpace(i, 10);
        EXPECT_EQ(i, mySpace.get_dimensions());
        EXPECT_EQ(10, mySpace.get_dimension_size());

        vector<int> point(i, i);
    
        EXPECT_EQ(0.0, mySpace.get(point));
        EXPECT_NO_THROW(mySpace.set(point, 23));
        EXPECT_EQ(23.0, mySpace.get(point));
    }
}

TEST(ErrorThrowing, TestsErrorThrowing){
    StateSpace mySpace(1, 5);

    EXPECT_THROW(mySpace.get({6}), std::out_of_range);

    EXPECT_THROW(mySpace.get({-1}), std::out_of_range);

    EXPECT_THROW(mySpace.get({5, 5}), std::invalid_argument);


    StateSpace mySpace2(4, 4);
    
    EXPECT_THROW(mySpace.get({1,1,1,5}), std::out_of_range);
    EXPECT_THROW(mySpace.set({1,1,1,5}, 1), std::out_of_range);

    EXPECT_THROW(mySpace.get({5, 5}), std::invalid_argument);

}