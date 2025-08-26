#ifndef MODEL_H
#define MODEL_H
#include <vector>

#include "../StateSpace/StateSpace.hpp"

class Model {
protected:
    StateSpace* stateSpace = nullptr;
    int totalQueries = 0;
    int currentQuery = 0;
public:
        Model(int dimensions, int dimensionSize, int totalQueries): stateSpace(new StateSpace(dimensions, dimensionSize)), totalQueries(totalQueries) {};
        virtual std::vector<int> get_next_query() = 0;
        virtual void update_prediction(const std::vector<int> &query, double result) = 0;
        StateSpace get_state_space() const { return *stateSpace; };
};


#endif // MODEL_H