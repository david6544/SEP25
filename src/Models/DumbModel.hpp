#if defined(DUMB) || defined(TESTING)
#ifndef DUMB_MODEL_H
#define DUMB_MODEL_H

#include "Model.hpp"

/**
 * @brief The DumbModel is merely for testing the client - It 
 * randomly picks query points and sets statespace[query point] = returned value
 * 
 */
class DumbModel : public Model {
public:
    DumbModel(int dimensions, int dimensionSize, int totalQueries);
    std::vector<int> get_next_query() override;
    void update_prediction(const std::vector<int> &query, double result) override;
};


#endif // DUMB_MODEL_H
#endif // DUMB || TESTING