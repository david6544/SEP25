#include "../src/Models/LinearModel.hpp"
#include "FunctionSpace.hpp"
#include "StateSpaceIO.hpp"

#include <iostream>
using namespace std;

double ackleyFunction(const std::vector<int>& query){
    const double a = 20.0;
    const double b = 0.2;
    const double c = 2 * M_PI;

    size_t n = query.size();
    if (n == 0) return 0.0;

    double sumSquares = 0.0;
    double sumCos = 0.0;

    for (int x : query) {
        double xd = static_cast<double>(x);  // convert int to double
        sumSquares += xd * xd;
        sumCos += cos(c * xd);
    }

    double term1 = -a * exp(-b * sqrt(sumSquares / n));
    double term2 = -exp(sumCos / n);

    return term1 + term2 + a + exp(1);
}

int main() {

    int dimensions = 1, dimensionSize = 100, queries = 70;
    FunctionSpace fspace(dimensions, dimensionSize, ackleyFunction);
    StateSpaceIO::set_IO(fspace);

    InputOutput* io = InputOutput::get_instance();
    LinearModel model(dimensions, dimensionSize, queries);

    for (int i = 0; i < queries; i++){
        std::vector<int> query = model.get_next_query();
        double result = io->send_query_recieve_result(query);
        model.update_prediction(query, result);
    }
    model.update_prediction_final();
    auto raw_state = model.get_state_space();
    io->output_state(raw_state);
}