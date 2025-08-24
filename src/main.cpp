#include <vector>
#include <string>
#include <iostream>

#include "InputOutput/CommandLineInputOutput.hpp"
#include "Models/DumbModel.hpp"
#include "Models/LinearModel.hpp"
#include "Models/RBFModel.hpp"

// --- Algorithms ---
void linearAlgorithm(int dimensions, int dimensionSize, int queries){
    InputOutput *io = InputOutput::get_instance();

    LinearModel model(dimensions, dimensionSize, queries);

    for (int i = 0; i < queries; i++){
        std::vector<int> query = model.get_next_query();
        double result = io->send_query_recieve_result(query);
        model.update_prediction(query, result);
    }
    model.update_prediction_final();
    const auto raw_state = model.get_state_space();
    io->output_state(raw_state);
}

void dumbAlgorithm(int dimensions, int dimensionSize, int queries){
    InputOutput *io = InputOutput::get_instance();

    DumbModel model(dimensions, dimensionSize);

    for (int i = 0; i < queries; i++){
        std::vector<int> query = model.get_next_query();
        double result = io->send_query_recieve_result(query);
        model.update_prediction(query, result);
    }
    const auto raw_state = model.get_state_space();
    io->output_state(raw_state);
}

void rbfAlgorithm(int dimensions, int dimensionSize, int queries){
    InputOutput *io = InputOutput::get_instance();

    RBFModel model(dimensions, dimensionSize);

    for (int i = 0; i < queries; i++){
        std::vector<int> query = model.get_next_query();
        double result = io->send_query_recieve_result(query);
        model.update_prediction(query, result);
    }
    model.update_prediction_final(); 
    const auto raw_state = model.get_state_space();
    io->output_state(raw_state);
}

// --- Main ---
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " Dimensions(int) ArraySize(int) MaxQueries(int) [modelType]\n"
                  << "modelType options: dumb | linear | rbf (default=linear)\n";
        return 1;
    }

    int dimensions = std::atoi(argv[1]);
    int dimensionSize = std::atoi(argv[2]);
    int queries = std::atoi(argv[3]);
    std::string modelType = (argc >= 5 ? argv[4] : "linear");

    CommandLineInputOutput::set_IO();

    if (modelType == "dumb") {
        dumbAlgorithm(dimensions, dimensionSize, queries);
    } else if (modelType == "rbf") {
        rbfAlgorithm(dimensions, dimensionSize, queries);
    } else {
        linearAlgorithm(dimensions, dimensionSize, queries);
    }
    // eg. run like this ./main 2 5 10 rbf

    return 0;
}
