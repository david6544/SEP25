/**
 * @file InputOutput.hpp
 * @author your name (you@domain.com) PLZ UPDATE
 * @brief This file declares an abstract InputOutput class that serves as a base for different input/output strategies.
 * @version 0.1
 * @date 2025-08-27
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#ifndef INPUT_OUTPUT_H
#define INPUT_OUTPUT_H
#include <vector>
#include "../StateSpace/StateSpace.hpp"

class InputOutput {
protected:
    static InputOutput* instance;
    InputOutput();
public:
    virtual double send_query_recieve_result(const std::vector<int> &query) = 0;
    virtual void output_state(const StateSpace &stateSpace) = 0;
    
    static InputOutput* get_instance();
};

#endif // INPUT_OUTPUT_H