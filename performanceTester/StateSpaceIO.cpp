#include "StateSpaceIO.hpp"

#include <iostream>
#include <filesystem>
#include <fstream>
#include <random>

using std::cout, std::endl;

FunctionSpace* StateSpaceIO::stateSpace = nullptr;
std::string StateSpaceIO::name = "";
int StateSpaceIO::queries = 0;
bool StateSpaceIO::stochastic = false;

void StateSpaceIO::set_IO(FunctionSpace& stateSpace, const std::string& name, int queries, bool stochastic){
    StateSpaceIO::stateSpace = &stateSpace;
    StateSpaceIO::name = name;
    StateSpaceIO::queries = queries;
    StateSpaceIO::stochastic = stochastic;
    if (instance == nullptr)
        instance = new StateSpaceIO;
}

void StateSpaceIO::set_state_space(FunctionSpace& stateSpace, const std::string& name, int queries){
    StateSpaceIO::stateSpace = &stateSpace;
    StateSpaceIO::name = name;
    StateSpaceIO::queries = queries;
}

static std::default_random_engine gen(std::random_device{}());
static std::uniform_real_distribution<double> dist(0.0, 1.0);
double StateSpaceIO::send_query_recieve_result(const std::vector<int> &query) {
    double result = stateSpace->get(query);
    if (stochastic) return dist(gen) <= result;
    return result;
}

void StateSpaceIO::output_state(Model &model, bool outputStateSpace) {
    stateSpace->resetResults();

    const int dimensions = model.get_dimensions();
    const int dimensionSize = model.get_dimensionSize();

    long long maxIdx = 1;
    for (int i = 0; i < dimensions; ++i)
        maxIdx *= dimensionSize;

    // Construct directory path
    std::string dir = "PerformanceData" +
                      std::to_string(dimensions) + ":" +
                      std::to_string(dimensionSize) + "/" +
                      this->name;

    std::filesystem::create_directories(dir);

    // Define file paths
    const std::string functionFile = dir + "/" + this->name + ".txt";
    const std::string queriesFile  = dir + "/" + this->name + "-" + std::to_string(this->queries) + ".txt";

    // Prepare output streams
    std::ofstream funcOut;
    const bool writeFunctionFile = !std::filesystem::exists(functionFile);
    if (writeFunctionFile) {
        funcOut.open(functionFile);
        if (!funcOut.is_open())
            throw std::runtime_error("Failed to open " + functionFile);
    }

    // Always recreate queries file
    std::ofstream queryOut(queriesFile, std::ios::trunc);
    if (!queryOut.is_open())
        throw std::runtime_error("Failed to open " + queriesFile);

    // Core write loop
    for (long long i = 0; i < maxIdx; ++i) {
        const auto coords = index_to_coords(i, dimensions, dimensionSize);
        const double funcVal = this->stateSpace->get(coords);
        const double queryVal = model.get_value_at(coords);

        stateSpace->updateResults(queryVal, funcVal);

        if (outputStateSpace){
            if (i > 0) {
                if (writeFunctionFile) funcOut << " ";
                queryOut << " ";
            }

            if (writeFunctionFile) funcOut << funcVal;
            queryOut << queryVal;
        }
    }

    if (writeFunctionFile) funcOut << std::endl;
    queryOut << std::endl;

    // Streams auto-close via RAII, but close explicitly for clarity
    if (writeFunctionFile) funcOut.close();
    queryOut.close();

    stateSpace->submitResults();
}

void StateSpaceIO::output_state(Model &model){
    output_state(model, false);
}


