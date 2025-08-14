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
#include <functional>
#include <typeinfo>

using namespace std;

namespace test {
    // allows for testing if two parameters are equal, parameters must be of the same type
    template <typename T>
    void assertEqual(const T& a, const T& b, const string message) {
        if (!(a == b)) {
            throw invalid_argument(string("Assertion failed: ") + message);
        }
    }

    // allows for testing if two parameters are not equal, parameters must be of the same type
    template <typename T>
    void assertNotEqual(const T& a, const T& b, const string message) {
        if (a == b) {
            throw invalid_argument("Assertion failed: " + message);
        }
    }
    /* allows for checking if a lambda function will throw a specific error
        func: A lambda function, example: [&myClass](){myClass.do(thing);} 
        this function is passed into assertThrow and executed inside the try catch.

        expectedExceptionType: The type of the exception that we expect to see, example: typeid(std::invalid_argument)
    */
    void assertThrow(function<void()> func, const std::type_info& expectedExceptionType, const string message) {
        try {
            func();
        } catch (const std::exception& e) {
            if (typeid(e) == expectedExceptionType) {
                return; 
            } else {
                throw invalid_argument("Assertion failed: "+ message + "\nWrong Exception: \n" + e.what());
            }
        } catch (...) {
            throw invalid_argument("Assertion failed: "+ message + " unknown was thrown");
        }

        throw invalid_argument("Assertion failed: "+ message + " nothing was flung");
    }

    /* allows for tests to be run and results to be output to the console
        example use: doTest(myFunctionName, "myFunctionName")
        testFunction: the testfunction, as in the example it is just the name of the test method 
        (test functions must be void functions)

        testName: should also just be the name of the method
    */
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