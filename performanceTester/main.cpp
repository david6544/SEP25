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

#include <iomanip>
#include <map>
#include <chrono>
struct PerfResult {
    std::string name;
    Results results;
};


PerfResult runPerfTest(int dimensions, int dimensionSize, int queries, SpaceFunctionType func, const std::string& name) {
    
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
    return {name, res};
}

// Run model against all functions in testfunctions and output a table
void runAllFunctions(int dimensions, int dimensionSize) {
    struct FuncInfo {
        SpaceFunctionType func;
        std::string name;
        std::string category;
    };
    std::vector<FuncInfo> funcs = {
        {testfunctions::ackleyFunction, "Ackley", "Many Local Minima"},
        {testfunctions::sumpow, "SumPow", "Bowl Shaped"},
        {testfunctions::griewank, "Griewank", "Many Local Minima"},
        {testfunctions::rastrigin, "Rastrigin", "Many Local Minima"},
        {testfunctions::michalewicz, "Michalewicz", "Steep Ridges/Drops"},
        {testfunctions::powerSum, "PowerSum", "Plate Function"},
        {testfunctions::zakharov, "Zakharov", "Plate Function"},
        {testfunctions::dixonPrice, "DixonPrice", "Valley Function"},
        {testfunctions::rosenbrock, "Rosenbrock", "Valley Function"},
        {testfunctions::hyperEllipsoid, "HyperEllipsoid", "Bowl Function"}
    };
    std::vector<double> queryPercents = {0.10, 0.25, 0.50, 0.75};
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\nPerformance Table:\n";
    std::cout << "---------------------------------------------------------------------------------------------------------------------------------------------------------------\n";
    std::cout << "| Function         | Category            |  Dim |  Size |  % Query | % Correct |    MAE       |     RMSE     |    Real Mean  |  Mean Predicted | Duration (s) |\n";
    std::cout << "---------------------------------------------------------------------------------------------------------------------------------------------------------------\n";
    // Store results for summary
    struct StatRow {
        std::string category;
        double percentCorrect = 0.0;
        double mae = 0.0;
        double rmse = 0.0;
        int count = 0;
    };
    std::map<std::string, StatRow> stats;
    #include <chrono>
    for (double percent : queryPercents) {
        for (const auto& f : funcs) {
            int totalArea = 1;
            for (int d = 0; d < dimensions; ++d) totalArea *= dimensionSize;
            int queries = std::max(1, static_cast<int>(totalArea * percent));
            auto start = std::chrono::high_resolution_clock::now();
            PerfResult r = runPerfTest(dimensions, dimensionSize, queries, f.func, f.name);
            auto end = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration<double>(end - start).count();
            std::cout << "| " << std::setw(16) << f.name << " | "
                      << std::setw(19) << f.category << " | "
                      << std::setw(4) << dimensions << " | "
                      << std::setw(5) << dimensionSize << " | "
                      << std::setw(7) << std::fixed << std::setprecision(0) << percent * 100 << "% |" << std::fixed << std::setprecision(3)
                      << std::setw(10) << r.results.percentCorrect() << " | "
                      << std::setw(12) << r.results.meanAbsoluteError() << " | "
                      << std::setw(12) << r.results.rootMeanSquaredError() << " | "
                      << std::setw(14) << r.results.realMean << " | "
                      << std::setw(14) << r.results.meanPredicted() << " | "
                      << std::setw(12) << std::fixed << std::setprecision(3) << duration << " |\n";
            // Collect stats
            auto& stat = stats[f.category];
            stat.category = f.category;
            stat.percentCorrect += r.results.percentCorrect();
            stat.mae += r.results.meanAbsoluteError();
            stat.rmse += r.results.rootMeanSquaredError();
            stat.count++;
        }
    }

    std::cout << "---------------------------------------------------------------------------------------------------------------------------------------------------------------\n";
    // Print summary table
    std::cout << "\nModel Performance by Function Category:\n";
    std::cout << "-----------------------------------------------------------------\n";
    std::cout << "| Category            | Avg % Correct |  Avg MAE   |  Avg RMSE  |\n";
    std::cout << "-----------------------------------------------------------------\n";
    for (const auto& [cat, stat] : stats) {
        std::cout << "| " << std::setw(19) << cat << " | "
                  << std::setw(13) << std::fixed << std::setprecision(3) << (stat.count ? stat.percentCorrect/stat.count : 0.0) << " | "
                  << std::setw(10) << (stat.count ? stat.mae/stat.count : 0.0) << " | "
                  << std::setw(10) << (stat.count ? stat.rmse/stat.count : 0.0) << " |\n";
    }
    std::cout << "-----------------------------------------------------------------\n";
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
    int dimensions = 2, dimensionSize = 100;
    testfunctions::dimSize = dimensionSize;
    runAllFunctions(dimensions, dimensionSize);
}
