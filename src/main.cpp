#include <vector>

#include "InputOutput/CommandLineInputOutput.hpp"
#include "Models/DumbModel.hpp"


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

int main(int argc, char* argv[]) {
    if (argc != 4) { // program name + 3 integers
        std::cerr << "Usage: " << argv[0] << " Dimensions : int,  Array size : int,  Maximum number of queries : int\n";
        return 1;
    }

    int dimensions = std::atoi(argv[1]);
    int dimensionSize = std::atoi(argv[2]);
    int queries = std::atoi(argv[3]);

    CommandLineInputOutput::set_IO();

    dumbAlgorithm(dimensions, dimensionSize, queries);

    return 0;
}
