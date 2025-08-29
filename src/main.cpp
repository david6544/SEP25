#include <vector>

#include "InputOutput/CommandLineInputOutput.hpp"

#if defined(LINEAR)
    #include "Models/LinearModel.hpp"
    #define CurrentModel LinearModel
#elif defined(DUMB)
    #include "Models/DumbModel.hpp"
    #define CurrentModel DumbModel
#else
    #error "Algorthim was not defined please check readme for build instructions"
#endif

void algorithm(int dimensions, int dimensionSize, int totalQueries){
    InputOutput *io = InputOutput::get_instance();

    CurrentModel model(dimensions, dimensionSize, totalQueries);

    for (int i = 0; i < totalQueries; i++){
        std::vector<int> query = model.get_next_query();
        double result = io->send_query_recieve_result(query);
        model.update_prediction(query, result);
    }
    const auto raw_state = model.get_state_space();
    io->output_state(raw_state);
}

int main(int argc, char* argv[]) {
    if (argc != 4) { // program name + 3 integers
        std::cerr << "Usage: " << argv[0] << " Dimensions : int,  Array size : int,  Maximum number of totalQueries : int\n";
        return 1;
    }
    
    int dimensions = std::atoi(argv[1]);
    int dimensionSize = std::atoi(argv[2]);
    int totalQueries = std::atoi(argv[3]);

    CommandLineInputOutput::set_IO();

    algorithm(dimensions, dimensionSize, totalQueries);

    return 0;
}
