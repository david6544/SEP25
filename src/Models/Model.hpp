#ifndef MODEL_H
#define MODEL_H
#include <vector>

#include "../StateSpace/StateSpace.hpp"

class Model {
protected:
    StateSpace* stateSpace = nullptr;
public:
        Model(int dimensions, int dimensionSize): stateSpace(new StateSpace(dimensions, dimensionSize)) {};
        virtual std::vector<int> get_next_query() = 0;
        virtual void update_prediction(std::vector<int> query, double result) = 0;
        StateSpace get_state_space() const { return *stateSpace; };
};


#endif // MODEL_H