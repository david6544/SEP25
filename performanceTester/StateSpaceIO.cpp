#include "StateSpaceIO.hpp"
#include <iostream>
using std::cout, std::endl;

FunctionSpace* StateSpaceIO::stateSpace = nullptr;

void StateSpaceIO::set_IO(FunctionSpace& stateSpace){
    if (instance == nullptr)
        StateSpaceIO::stateSpace = &stateSpace;
        instance = new StateSpaceIO;
}

void StateSpaceIO::set_state_space(FunctionSpace& stateSpace){
    StateSpaceIO::stateSpace = &stateSpace;
}

double StateSpaceIO::send_query_recieve_result(const std::vector<int> &query) {
    double result = stateSpace->get(query);
    return result;
}

void StateSpaceIO::output_state(StateSpace &predictedStateSpace) {
    Results results = stateSpace->getResults(predictedStateSpace);

    cout << "Performance Metrics:" << endl;
    cout << "-------------------" << endl;
    cout << "Percent Correct: " << results.percentCorrect() << "%" << endl;
    cout << "Mean Absolute Error: " << results.meanAbsoluteError() << endl;
    cout << "Root Mean Squared Error: " << results.rootMeanSquaredError() << endl;
    cout << "Mean Predicted Value: " << results.meanPredicted() << endl;
    cout << "Actual Min/Max: [" << results.minActual << ", " << results.maxActual << "]" << endl;
    cout << "Predicted Min/Max: [" << results.minPredicted << ", " << results.maxPredicted << "]" << endl;
}