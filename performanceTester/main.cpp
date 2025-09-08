
#if defined(LINEAR)
    #include "../src/Models/LinearModel.hpp"
    #define CurrentModel LinearModel
#elif defined(DUMB)
    #include "../src/Models/DumbModel.hpp"
    #define CurrentModel DumbModel
#elif defined(RBF)
    #include "../src/Models/RBFModel.hpp"
    #define CurrentModel RBFModel
#else
    #error "Algorthim was not defined please check readme for build instructions"
#endif

#include "FunctionList.hpp"
#include "FunctionSpace.hpp"
#include "StateSpaceIO.hpp"

#include <iostream>

#include <iomanip>
#include <chrono>
// Helper to run model on a given function and return Results and real mean
struct PerfResult {
    std::string name;
    Results results;
    double timeTaken;
};

PerfResult runPerfTest(int dimensions, int dimensionSize, int queries, SpaceFunctionType func, const std::string& name) {
    auto start = std::chrono::high_resolution_clock::now();
    FunctionSpace fspace(dimensions, dimensionSize, func);
    StateSpaceIO::set_IO(fspace);
    InputOutput* io = InputOutput::get_instance();
    CurrentModel model(dimensions, dimensionSize, queries);
    for (int i = 0; i < queries; i++) {
        std::vector<int> query = model.get_next_query();
        double result = io->send_query_recieve_result(query);
        model.update_prediction(query, result);
    }
    Results res = fspace.getResults(model);
    auto end = std::chrono::high_resolution_clock::now();
    double timeTaken = std::chrono::duration<double>(end - start).count();
    return {name, res, timeTaken};
}

// Run model against all functions in testfunctions and output a table
void runAllFunctions(int dimensions, int dimensionSize) {
    struct FuncInfo {
        SpaceFunctionType func;
        std::string name;
    };
    std::vector<FuncInfo> funcs = {
        {testfunctions::ackleyFunction, "Ackley"},
        {testfunctions::sumpow, "SumPow"},
        {testfunctions::griewank, "Griewank"},
        {testfunctions::rastrigin, "Rastrigin"},
        {testfunctions::michalewicz, "Michalewicz"}
    };
    std::vector<double> queryPercents = {0.10, 0.25, 0.50, 0.75};
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\nPerformance Table:\n";
    std::cout << "------------------------------------------------------------------------------------------------------------------------\n";
    std::cout << "| Function     |  Dim |  Size  | % Query | % Correct |  MAE      |  RMSE     |  Real Mean | Mean Predicted | Time (s) |\n";
    std::cout << "------------------------------------------------------------------------------------------------------------------------\n";
    for (double percent : queryPercents) {
        for (const auto& f : funcs) {
            int totalArea = 1;
            for (int d = 0; d < dimensions; ++d) totalArea *= dimensionSize;
            int queries = std::max(1, static_cast<int>(totalArea * percent));
            PerfResult r = runPerfTest(dimensions, dimensionSize, queries, f.func, f.name);
            std::cout << "| " << std::setw(12) << r.name << " | "
                      << std::setw(4) << dimensions << " | "
                      << std::setw(5) << dimensionSize << " | "
                      << std::setw(5) << std::fixed << std::setprecision(0) << percent * 100 << "% |" << std::fixed << std::setprecision(3)
                      << std::setw(9) << r.results.percentCorrect() << " | "
                      << std::setw(9) << r.results.meanAbsoluteError() << " | "
                      << std::setw(9) << r.results.rootMeanSquaredError() << " | "
                      << std::setw(10) << r.results.realMean << " | "
                      << std::setw(11) << r.results.meanPredicted() << " | "
                      << std::setw(9) << r.timeTaken << " |\n";
        }
    }
    std::cout << "-------------------------------------------------------------------------------------------------------------\n";
}

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
 * This function is the main driver for performance testing, set the dimension size
 * It will run your model against every function in the function suites, 
 * along with a percentage of the queries of any given dimension size
 * 
 */
int main(void) {
    int dimensions = 3, dimensionSize = 10;
    runAllFunctions(dimensions, dimensionSize);
}
