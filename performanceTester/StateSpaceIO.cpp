#include "StateSpaceIO.hpp"

#include <iostream>
#include <filesystem>
#include <fstream>

using std::cout, std::endl;

FunctionSpace* StateSpaceIO::stateSpace = nullptr;
std::string StateSpaceIO::name = "";
int StateSpaceIO::queries = 0;

void StateSpaceIO::set_IO(FunctionSpace& stateSpace, const std::string& name, int queries){
    StateSpaceIO::stateSpace = &stateSpace;
    StateSpaceIO::name = name;
    StateSpaceIO::queries = queries;
    if (instance == nullptr)
        instance = new StateSpaceIO;
}

void StateSpaceIO::set_state_space(FunctionSpace& stateSpace, const std::string& name, int queries){
    StateSpaceIO::stateSpace = &stateSpace;
    StateSpaceIO::name = name;
    StateSpaceIO::queries = queries;
}

double StateSpaceIO::send_query_recieve_result(const std::vector<int> &query) {
    double result = stateSpace->get(query);
    return result;
}

void StateSpaceIO::output_state(Model &model) {
    int dimensions = model.get_dimensions();
    int dimensionSize = model.get_dimensionSize();

    long long maxIdx = 1;
    for (int i = 0; i < dimensions; ++i)
        maxIdx *= dimensionSize;

    // Base directory
    std::string dir = "PerformanceData" + 
                    std::to_string(dimensions)+
                    ":"+
                    std::to_string(dimensionSize)
                    +"/" + this->name;

    std::filesystem::create_directories(dir);

    // File paths
    std::string functionFile = dir + "/" + this->name + ".txt";
    std::string queriesFile  = dir + "/" + this->name + "-" + std::to_string(this->queries) + ".txt";

    if (!std::filesystem::exists(functionFile)) {
        std::ofstream out(functionFile);
        if (!out.is_open()) {
            throw std::runtime_error("Failed to open " + functionFile);
        }

        for (long long i = 0; i < maxIdx; i++) {
            auto coords = index_to_coords(i, dimensions, dimensionSize);
            double funcVal = this->stateSpace->get(coords);
            
            if (i > 0) out << " ";
            out << funcVal;
        }
        out << std::endl;
        out.close();
    } 

    std::ofstream out(queriesFile, std::ios::app);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open " + queriesFile);
    }

    for (long long i = 0; i < maxIdx; i++) {
        auto coords = index_to_coords(i, dimensions, dimensionSize);
        double stateVal = model.get_value_at(coords);

        if (i > 0) out << " ";
        out << stateVal;
    }
    out << std::endl;
    out.close();
}
