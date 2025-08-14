/* 

testing.hpp

provides tooling for testing

*/


#ifndef TESTING_H
#define TESTING_H
#include <vector>
#include <iostream>
#include <cmath>
#include <string>

using namespace std;

namespace test {
    template <typename T>
    void assertEqual(const T& a, const T& b, const string message) {
        if (!(a == b)) {
            throw invalid_argument(string("Assertion failed: ") + message);
        }
    }

    template <typename T>
    void assertNotEqual(const T& a, const T& b, const string message) {
        if (a == b) {
            throw invalid_argument("Assertion failed: " + message);
        }
    }



    void doTest(void (*testFunction)(void), string testName){
        cout << "Starting " << testName << "........";
        try {
            testFunction();
        } catch (const exception& e){
            cout << "Failed: failed with" << endl << e.what() << endl;
            return;
        } catch (...) {
            cout << "Failed: failed with unknown exception type" << endl;
            return;
        }

        cout << "Success: completed successfully" << endl;    
    }
    
}

#endif // TESTING_H