#if defined(LINEAR)
    #include "../src/Models/LinearModel.hpp"
    #define CurrentModel LinearModel
#elif defined(DUMB)
    #include "../src/Models/DumbModel.hpp"
    #define CurrentModel DumbModel
#elif defined(TestM)
    #include "../src/Models/TestModel.hpp"
    #define CurrentModel TestModel
#elif defined(RBF)
    #include "../src/Models/RBF.hpp"
    #define CurrentModel RBFModel
#elif defined(STOCTREE)
    #include "../src/Models/StochasticQueryModel.hpp"
    #define CurrentModel StochasticQueryModel
#elif defined(GEK)
    #include "../src/Models/GEKModel.hpp"
    #define CurrentModel GEKModel
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


PerfResult runPerfTest(int dimensions, int dimensionSize, bool outputStateSpace, bool stochastic, int queries, SpaceFunctionType func, const std::string& name) {
    
    FunctionSpace fspace(dimensions, dimensionSize, func);
    
    StateSpaceIO::set_IO(fspace, name, queries, stochastic);
    StateSpaceIO* io = (StateSpaceIO*) InputOutput::get_instance();

    CurrentModel model(dimensions, dimensionSize, queries);
    
    for (int i = 0; i < queries; i++) {
        std::vector<int> query = model.get_next_query();

        double result = io->send_query_recieve_result(query);

        model.update_prediction(query, result);
    }
    
    io->output_state(model, outputStateSpace);

    return {name, fspace.getAllResults().back()};
}

// Run model against all functions in testfunctions and output a table
void runAllFunctions(int dimensions, int dimensionSize, bool outputStateSpace, bool stochastic) {
    
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
    std::vector<double> queryPercents = {0.3};
    
    if (stochastic) {
        // queryPercents = {0.3};
        queryPercents = {0.3, 1.5, 2};
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\nPerformance Table:\n";
    std::cout << "---------------------------------------------------------------------------------------------------------------------------------------------------------------\n";
    std::cout << "| Function         | Category            |  Dim |  Size |  % Query | % Correct |    MAE       |     RMSE     |    Real Mean  |  Mean Predicted | Duration (s) |\n";
    std::cout << "---------------------------------------------------------------------------------------------------------------------------------------------------------------\n";
    // Store results for summary
    struct StatRow {
        std::string category;
        std::vector<double> percentCorrects;
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
            PerfResult r = runPerfTest(dimensions, dimensionSize, outputStateSpace, stochastic, queries, f.func, f.name);
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
                      << std::setw(14) << r.results.meanActual() << " | "
                      << std::setw(14) << r.results.meanPredicted() << " | "
                      << std::setw(12) << std::fixed << std::setprecision(3) << duration << " |\n";
            // Collect stats
            auto& stat = stats[f.category];
            stat.category = f.category;
            stat.percentCorrects.push_back(r.results.percentCorrect());
            stat.mae += r.results.meanAbsoluteError();
            stat.rmse += r.results.rootMeanSquaredError();
            stat.count++;
        }
    }

    std::cout << "---------------------------------------------------------------------------------------------------------------------------------------------------------------\n";
    // Print summary table
    std::cout << "\nModel Performance by Function Category and Query Size:\n";
    std::cout << "-----------------------------------------------------------------------------------------------\n";
    std::cout << "| Category            | 10% Query Space | 25% Query Space | 50% Query Space | 75% Query Space |\n";
    std::cout << "-----------------------------------------------------------------------------------------------\n";
    for (const auto& [cat, stat] : stats) {
        std::cout << "| " << std::setw(19) << cat << " | ";
        for (size_t i = 0; i < 4; ++i) {
            double avg = 0.0;
            int n = stat.percentCorrects.size() / 4;
            if (n > 0) {
                double sum = 0.0;
                for (size_t j = i; j < stat.percentCorrects.size(); j += 4) {
                    sum += stat.percentCorrects[j];
                }
                avg = sum / n;
            }
            std::cout << std::setw(15) << std::fixed << std::setprecision(3) << avg << " | ";
        }
        std::cout << "\n";
    }
    std::cout << "-----------------------------------------------------------------------------------------------\n";
}

/**
 * This function is the main driver for performance testing, set the dimension size
 * It will run your model against every function in the function suites, 
 * along with a percentage of the queries of any given dimension size
 * 
 */
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] 
                  << " <dimensions> <dimensionSize> [--output shouldOutput] [--rand testStochastic]\n";
        return 1;
    }

    int dimensions = std::atoi(argv[1]);
    int dimensionSize = std::atoi(argv[2]);

    testfunctions::minMaxCache.clear();
    testfunctions::dimSize = dimensionSize;
    testfunctions::dims = dimensions;

    bool outputStateSpace = false;
    bool stochastic = false;

    for (int i = 3; i < argc; i++){
        std::string opt = argv[i];
        if (opt == "--output" && i + 1 < argc){
            opt = argv[i+1];
            if (opt == "1" || opt == "true" || opt == "yes") {
                outputStateSpace = true;
                i++;
            }
        }

        if (opt == "--rand" && i + 1 < argc){
            opt = argv[i+1];
            if (opt == "1" || opt == "true" || opt == "yes") {
                stochastic = true;
            }
        }
    }
    testfunctions::dimSize = dimensionSize;
    runAllFunctions(dimensions, dimensionSize, outputStateSpace, stochastic);
}
