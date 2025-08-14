#include <iostream>

#include "../StateSpace.hpp"
#include "../../tools/Testing.hpp"

using namespace std;

void test_one_dimension_access(){
    StateSpace mySpace(1, 5);
    test::assertEqual(0.0, mySpace.get({1}), "test access");
    test::assertEqual(0.0, mySpace.get({4}), "test access");
    mySpace.set({1}, -1);
    test::assertEqual(-1.0, mySpace.get({1}), "test updated");
}

void test_multiple_dimensions() {
    for (int i = 1; i < 6; i++){
        StateSpace mySpace(i, 10);
        test::assertEqual(i, mySpace.get_dimensions(), "dimension check");
        test::assertEqual(10, mySpace.get_dimension_size(), "dimension size check");

        vector<int> point(i, i);
    
        test::assertEqual(0.0, mySpace.get(point), "test access");
        mySpace.set(point, 23);
        test::assertEqual(23.0, mySpace.get(point), "test updated");
    }
}

void test_error_throwing(){
    StateSpace mySpace(1, 5);

    test::assertThrow([mySpace](){mySpace.get({6});}, 
                    typeid(std::out_of_range), 
                    "out_of_range exception was not thrown when it should have been");

    test::assertThrow([mySpace](){mySpace.get({5, 5});}, 
                    typeid(std::invalid_argument), 
                    "out_of_range exception was not thrown when it should have been");


    StateSpace mySpace2(4, 4);
    
    test::assertThrow([&mySpace2](){mySpace2.get({1,1,1,5});}, 
                    typeid(std::out_of_range), 
                    "out_of_range exception was not thrown when it should have been");

    test::assertThrow([&mySpace2](){mySpace2.set({1,1,1,5}, 1.0);}, 
                    typeid(std::out_of_range), 
                    "out_of_range exception was not thrown when it should have been");

    test::assertThrow([&mySpace2](){mySpace2.get({5, 5});}, 
                    typeid(std::invalid_argument), 
                    "out_of_range exception was not thrown when it should have been");

}

int main() {
    test::doTest(test_one_dimension_access, "test_one_dimension_access");

    test::doTest(test_multiple_dimensions, "test_multiple_dimensions");

    test::doTest(test_error_throwing, "test_error_throwing");
}