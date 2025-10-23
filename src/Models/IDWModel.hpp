#ifndef IDWMODEL_H
#define IDWMODEL_H

#include "Model.hpp"
#include "../StateSpace/KDTreeStateSpace.hpp"
#include <vector>

class IDWModel : public Model {
protected:
    int dimensions, dimensionSize, totalQueries, currentQuery = 0;
    KDTreeStateSpace* stateSpace;
public:
        IDWModel(int dimensions, int dimensionSize, int totalQueries);
        int get_dimensions(){ return dimensions; }
        int get_dimensionSize(){ return dimensionSize; }
        std::vector<int> get_next_query() override;
        void update_prediction(const std::vector<int> &query, double result) override;
        double get_value_at(const std::vector<int> &query) override;
};


#endif // IDWMODEL_H