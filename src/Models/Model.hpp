#ifndef MODEL_H
#define MODEL_H
#include <vector>

#include "../StateSpace/ArrayStateSpace.hpp"

class Model {
protected:
    ArrayStateSpace* stateSpace = nullptr;
    int totalQueries = 0;
    int currentQuery = 0;
public:
        Model(int dimensions, int dimensionSize, int queries): stateSpace(new ArrayStateSpace(dimensions, dimensionSize)), totalQueries(queries) {};
        virtual std::vector<int> get_next_query() = 0;
        virtual void update_prediction(const std::vector<int> &query, double result) = 0;
        ArrayStateSpace get_state_space() const { return *stateSpace; };
};


#endif // MODEL_H