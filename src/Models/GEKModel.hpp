#if defined(GEK) || defined(TESTING)
#ifndef GEK_MODEL_H
#define GEK_MODEL_H

#include "Model.hpp"
#include "Querying/QueryTree.hpp"
#include "Mapping/GEKMapping.hpp"
#include <memory>
#include <vector>

/**
 * @brief The GEKModel implements Gradient-Enhanced Kriging optimization.
 * 
 * It uses QueryTree for intelligent query selection and GEKMapping for
 * Gaussian process regression-based interpolation.
 */
class GEKModel : public Model {
public:
    GEKModel(int dimensions, int dimensionSize, int totalQueries);
    ~GEKModel();
    
    std::vector<int> get_next_query() override;
    void update_prediction(const std::vector<int> &query, double result) override;
    double get_value_at(const std::vector<int> &query) override;
    
    /**
     * @brief Configure which dimensions have periodic boundary conditions
     * 
     * @param periodic Vector of booleans indicating which dimensions are periodic
     */
    void set_periodic_dimensions(const std::vector<bool>& periodic);

private:
    QueryTree* queryTree;
    GEKMapping* mapping;
    std::vector<std::pair<double, std::vector<int>>> data;
    int leafSize;
    bool use_local_neighborhood;
    int local_k;
    std::vector<bool> periodic_dims;
};

#endif //GEK_MODEL_H
#endif // GEK || TESTING