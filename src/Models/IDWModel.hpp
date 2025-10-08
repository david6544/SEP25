#if defined(IDW) || defined(TESTING)
#ifndef IDW_MODEL_H
#define IDW_MODEL_H

#include "Model.hpp"
#include "../StateSpace/KDTreeStateSpace.hpp"

/**
 * @brief IDWModel implements Inverse Distance Weighting interpolation using a KD-Tree for spatial indexing.
 * 
 */ 
class IDWModel : public Model {
    KDTreeStateSpace* stateSpace;
    int visitCount;        // How many times queries have been made near this node
    double totalError;     // Cumulative prediction error at this location
    std::unordered_set<std::string> visited;

    double predictValueAt(const std::vector<int>& query);

public:
    IDWModel(int dimensions, int dimensionSize, int totalQueries);

    std::vector<int> get_next_query() override;
    
    void update_prediction(const std::vector<int> &query, double result) override;
    
    double get_value_at(const std::vector<int> &query) override;
};


#endif // IDW_MODEL_H
#endif // IDW || TESTING