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

std::vector<int> index_to_coords(int index, int dimensions, int dimensionSize) {
    std::vector<int> coords;
    coords.reserve(dimensions);
    long long multiplier = pow(dimensionSize, dimensions);
    for (int i = dimensions - 1; i > -1; i--) {
        coords.push_back(index % multiplier);
        if (coords.back() >= dimensionSize){
            throw std::out_of_range("computed index is out of range");
        }
        index /= multiplier;
        multiplier /= dimensionSize;
    }
    return coords;
}

void CommandLineInputOutput::output_state(Model &model){
    int dimensions = model.get_dimensions();
    int dimensionSize = model.get_dimensionSize();

    int maxIdx = pow(dimensionSize, dimensions);
    for (int i = 0; i < maxIdx; i++) {
        std::cout << model.get_value_at(index_to_coords(i, dimensions, dimensionSize)) << ((i == maxIdx-1) ? "\n" : " ");
    }
}