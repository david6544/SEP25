/**
 * @file CommandLineInputOutput.cpp
 * @author your name (you@domain.com) PLS UPDATE
 * @brief This file implements a CommandLineInputOutput class that handles input and output via the command line.
 * @version 0.1
 * @date 2025-08-27
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#include <iostream>
#include "CommandLineInputOutput.hpp"

void CommandLineInputOutput::set_IO(){
    if (instance == nullptr)
        instance = new CommandLineInputOutput;
}

double CommandLineInputOutput::send_query_recieve_result(const std::vector<int> &query){

    // output query to the CL
    for (int i = 0; i < query.size(); i++)
        std::cout << query[i] << ((i == query.size()-1) ? "\n" : ",");

    double result;
    std::cin >> result;

    return result;
}

void CommandLineInputOutput::output_state(const StateSpace &stateSpace){
    std::vector<double> vec2Output = stateSpace.get_raw_representation();
    if (vec2Output.size() == 0) return;
    for (int i = 0; i < vec2Output.size(); i++)
        std::cout << vec2Output[i] << ((i == vec2Output.size()-1) ? "\n" : " ");

}