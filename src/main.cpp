#include <vector>
#include <iostream>

#include "InputOutput/CommandLineInputOutput.hpp"

#if defined(LINEAR)
    #include "Models/LinearModel.hpp"
    #define CurrentModel LinearModel
#elif defined(DUMB)
    #include "Models/DumbModel.hpp"
    #define CurrentModel DumbModel
#elif defined(TestM)
    #include "Models/TestModel.hpp"
    #define CurrentModel TestModel
#elif defined(RBF)
    #include "Models/RBF.hpp"
    #define CurrentModel RBFModel
#elif defined(STOCTREE)
    #include "Models/StochasticQueryModel.hpp"
    #define CurrentModelStochasticQueryModel
#elif defined(GEK)
    #include "Models/GEKModel.hpp"
    #define CurrentModel GEKModel
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
    io->output_state(model);
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