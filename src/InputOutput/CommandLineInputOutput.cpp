#include <iostream>

#include "CommandLineInputOutput.hpp"

double CommandLineInputOutput::send_query_recieve_result(std::vector<int> &query){
    // output query to the CL
    for (int i = 0; i < query.size(); i++)
        std::cout << i << (i = query.size()-1) ? "\n" : ",";
    
    double result;
    std::cin >> result;

    return result;
}

void CommandLineInputOutput::output_state(StateSpace &stateSpace){
    return;
}