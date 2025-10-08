#if defined(LINEAR) || defined(TESTING)
#include <vector>
#include <random>
#include <ctime>

#include "LinearModel.hpp"

LinearModel::LinearModel(int dimensions, int dimensionSize, int totalQueries) : 
    Model(dimensions, dimensionSize, totalQueries), stateSpace(new ArrayStateSpace(dimensions, dimensionSize)) {
    std::srand(std::time(nullptr));
}

std::vector<int> LinearModel::get_next_query() {
    currentQuery++;

    static int traversed_ix = 0;

    int D = this->stateSpace->get_dimensions();
    int K = this->stateSpace->get_dimension_size();
    std::vector<int> nextQuery(D, 0);

    double step = ((double)(K-1))/(double)totalQueries;
    for (auto &var : nextQuery){
        double vard = step/2.0 + ((double)traversed_ix) * step;
        var = (int)std::round(vard);
        traversed_ix++;
        // std::cout << "vard is " << vard << "\n";
        // std::cout << "var  is " << var << "\n";
    }

    return nextQuery;
}

void print_statespace(ArrayStateSpace& stateSpace) {
    std::vector<double> vec2Output = stateSpace.get_raw_representation();
    if (vec2Output.size() == 0) return;
    for (int i = 0; i < vec2Output.size(); i++)
        std::cout << vec2Output[i] << ((i == vec2Output.size()-1) ? "\n" : " ");
}

int LinearModel::find_next_nonzero_ix(double ix) {
    int D = this->stateSpace->get_dimensions();
    int K = this->stateSpace->get_dimension_size();
    int traversed_ix = ix;
    while (this->stateSpace->get({traversed_ix}) == 0 && traversed_ix < K-1) {
        traversed_ix++;
    }
    // std::cout << "next nonzero " << traversed_ix << "\n";
    // print_statespace(*this->stateSpace);
    return traversed_ix;
}


int LinearModel::find_prev_nonzero_ix(double ix) {
    int D = this->stateSpace->get_dimensions();
    int K = this->stateSpace->get_dimension_size();
    int traversed_ix = ix;
    while (this->stateSpace->get({traversed_ix}) == 0 && traversed_ix > 0) {
        traversed_ix--;
    }
    // std::cout << "prev nonzero " << traversed_ix << "\n";
    // print_statespace(*this->stateSpace);
    return traversed_ix;
}

int LinearModel::find_next_nonzero_afterzero_ix(double ix) {
    int D = this->stateSpace->get_dimensions();
    int K = this->stateSpace->get_dimension_size();
    int traversed_ix = ix;
    while (this->stateSpace->get({traversed_ix}) != 0 && traversed_ix < K-1) {
        traversed_ix++;
    }
    while (this->stateSpace->get({traversed_ix}) == 0 && traversed_ix < K-1) {
        traversed_ix++;
    }
    // std::cout << "next nonzero " << traversed_ix << "\n";
    // print_statespace(*this->stateSpace);
    return traversed_ix;
}


int LinearModel::find_prev_nonzero_afterzero_ix(double ix) {
    int D = this->stateSpace->get_dimensions();
    int K = this->stateSpace->get_dimension_size();
    int traversed_ix = ix;
    while (this->stateSpace->get({traversed_ix}) != 0 && traversed_ix > 0) {
        traversed_ix--;
    }
    while (this->stateSpace->get({traversed_ix}) == 0 && traversed_ix > 0) {
        traversed_ix--;
    }
    // std::cout << "prev nonzero " << traversed_ix << "\n";
    // print_statespace(*this->stateSpace);
    return traversed_ix;
}

double LinearModel::get_value_at(const std::vector<int> &query) {
    return this->stateSpace->get(query);
}

void LinearModel::update_prediction_final() {
    
    int D = this->stateSpace->get_dimensions();
    int K = this->stateSpace->get_dimension_size();
    // std::cout << "D=" << D << " K=" << K << "\n";

    int traversed_ix = 0;
    int nonzero_ix = 0;
    int left_nonzero_ix = find_next_nonzero_ix(0);

    nonzero_ix = left_nonzero_ix;
    for (int i = 0; i < nonzero_ix; i++) {
        this->stateSpace->set({i}, this->stateSpace->get({nonzero_ix}));
    }

    nonzero_ix = find_prev_nonzero_ix(K-1);
    for (int i = K-1; i > nonzero_ix; i--) {
        this->stateSpace->set({i}, this->stateSpace->get({nonzero_ix}));
    }

    int right_nonzero_ix = find_next_nonzero_afterzero_ix(left_nonzero_ix);
    int round = 1;
    // print_statespace(*this->stateSpace);
    // std::cout << "Begin Loop:\n";
    while (right_nonzero_ix < K-1 && round < K * K + 10) {
        // std::cout << "Round: " << round << " (" << left_nonzero_ix << " - " << right_nonzero_ix << ")\n";
        // print_statespace(*this->stateSpace);
        if (left_nonzero_ix >= right_nonzero_ix-1) {
            // std::cout << "equal so moving...\n";
            right_nonzero_ix = find_next_nonzero_afterzero_ix(right_nonzero_ix);
            left_nonzero_ix = find_prev_nonzero_afterzero_ix(right_nonzero_ix);
        } else {
            int ix = left_nonzero_ix+1;
            double left_nonzero_val = this->stateSpace->get({left_nonzero_ix});
            double right_nonzero_val = this->stateSpace->get({right_nonzero_ix});
            double val = left_nonzero_val + (right_nonzero_val - left_nonzero_val) * (((double)(ix-left_nonzero_ix))/((double)(right_nonzero_ix-left_nonzero_ix)));
            this->stateSpace->set({ix}, val); 
            left_nonzero_ix++;
        }
        round++;
        // print_statespace(*this->stateSpace);
    }

}

void LinearModel::update_prediction(const std::vector<int> &query, double result) {
    stateSpace->set(query, result);
    if (currentQuery == totalQueries)
        update_prediction_final();
}

#endif // LINEAR || TESTING