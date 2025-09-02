
#if defined(LINEAR)
    #include "../src/Models/LinearModel.hpp"
    #define CurrentModel LinearModel
#elif defined(DUMB)
    #include "../src/Models/DumbModel.hpp"
    #define CurrentModel DumbModel
#else
    #error "Algorthim was not defined please check readme for build instructions"
#endif

#include "FunctionList.hpp"
#include "FunctionSpace.hpp"
#include "StateSpaceIO.hpp"

#include <iostream>


//This is gpt'd lmao,
void output_performance(std::vector<Results> results) {
    std::vector<double> errors;
    std::cout << "\n \n";

    for (const auto& res : results) {
        // Use mean absolute error as a representative error for each Results
        errors.push_back(res.meanAbsoluteError());
    }
    if (errors.empty()) {
        std::cout << "No error data available." << std::endl;
        return;
    }
    std::sort(errors.begin(), errors.end());
    double median;
    size_t n = errors.size();
    if (n % 2 == 0) {
        median = (errors[n/2 - 1] + errors[n/2]) / 2.0;
    } else {
        median = errors[n/2];
    }
    std::cout << "Median of mean absolute errors: " << median << std::endl;
}

void runSeveral(int dimensions, int dimensionSize, SpaceFunctionType func) {
    std::vector<double> querySizes = {0.10, 0.25, 0.50, 0.75};
    std::vector<Results> results;
    for (auto size: querySizes) {
        int queryCount = dimensionSize / size;
        
        FunctionSpace fspace(dimensions, dimensionSize, func);
        StateSpaceIO::set_IO(fspace);
        InputOutput* io = InputOutput::get_instance();
        
        CurrentModel model(dimensions, dimensionSize, queryCount);
        
        for (int i = 0; i < queryCount; i++){
            std::vector<int> query = model.get_next_query();
            double result = io->send_query_recieve_result(query);
            model.update_prediction(query, result);
        }
        results.emplace_back(fspace.getResults(model));
        io->output_state(model);
    }
    output_performance(results);

}

void runSingle(int dimensions, int dimensionSize, int queries, SpaceFunctionType func) {
    
    FunctionSpace fspace(dimensions, dimensionSize, func);
    StateSpaceIO::set_IO(fspace);
    InputOutput* io = InputOutput::get_instance();
    
    CurrentModel model(dimensions, dimensionSize, queries);
    
    for (int i = 0; i < queries; i++){
        std::vector<int> query = model.get_next_query();
        double result = io->send_query_recieve_result(query);
        model.update_prediction(query, result);
    }
    io->output_state(model);
}

/**
 * This function is the main driver for performance testing, set the paramters below,
 *  pick a function and either run a single test, or one with a variable number of query sizes
 * 
 *  Note: RunSeveral doens't output much inisghtful data just yet
 */
int main(void) {

    int dimensions = 2, dimensionSize = 600, queries = 6000;
    auto func = testfunctions::rastrigin;
    //runSeveral(dimensions,dimensionSize, func);
    runSingle(dimensions, dimensionSize, queries, func);
}
