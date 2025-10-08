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
#include <cmath>

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

void CommandLineInputOutput::output_state(Model &model){
    int dimensions = model.get_dimensions();
    int dimensionSize = model.get_dimensionSize();

    long long maxIdx = 1;
    for (int i = 0; i < dimensions; ++i)
        maxIdx *= dimensionSize;

    auto coords = index_to_coords(0, dimensions, dimensionSize);
    std::cout << model.get_value_at(coords);
    for (long long i = 1; i < maxIdx; i++) {
        coords = index_to_coords(i, dimensions, dimensionSize);
        std::cout << " " << model.get_value_at(coords);
    }
    std::cout << std::endl;
}