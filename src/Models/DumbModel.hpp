#ifndef DUMB_MODEL_H
#define DUMB_MODEL_H

#include "Model.hpp"

class DumbModel : public Model {
public:
    DumbModel(int dimensions, int dimensionSize);
    std::vector<int> get_next_query() override;
    void update_prediction(const std::vector<int> &query, double result) override;
};


#endif //DUMB_MODEL_H