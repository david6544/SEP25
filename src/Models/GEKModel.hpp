#if defined(GEK) || defined(TESTING)
#ifndef GEK_MODEL_H
#define GEK_MODEL_H

#include "Model.hpp"
#include "Querying/QueryTree.hpp"
#include "Mapping/GEKMapping.hpp"
#include <memory>
#include <vector>

/**
 * @file GEKModel.hpp
 * @brief Gradient-Enhanced Kriging (GEK) Model wrapper.
 *
 * @details
 * High-level overview:
 * The GEKModel provides an experiment-facing interface for using Gradient-
 * Enhanced Kriging interpolation (implemented in `GEKMapping`) together with a
 * query-selection strategy (`QueryTree`). It issues queries via
 * `get_next_query()`, collects observed values via `update_prediction(...)`,
 * and returns estimates with `get_value_at(...)`.
 *
 * Relation to the state-space mapping challenge:
 * - GEK is a regression-based approach (Gaussian process-like) which can
 *   exploit gradient/structure to provide smooth interpolations and uncertainty
 *   estimates.
 * - It complements grid/interpolation approaches (Bilinear/Region models) by
 *   offering a learned mapping that can capture complex correlations if enough
 *   data is available.
 * - Periodic boundary conditions can be configured to handle wrapped axes.
 *
 * Public contract: construct the model, call `get_next_query()` repeatedly to
 * receive sampling coordinates, pass observed results to
 * `update_prediction(...)`, and query estimates with `get_value_at(...)`.
 */

class GEKModel : public Model {
public:
    /**
     * @brief Construct a new GEKModel
     *
     * @param dimensions number of spatial dimensions
     * @param dimensionSize size of each dimension
     * @param totalQueries total query budget for the experiment
     */
    GEKModel(int dimensions, int dimensionSize, int totalQueries);

    /**
     * @brief Destroy the GEKModel
     */
    ~GEKModel();
    
    /**
     * @brief Return the next coordinate to query.
     *
     * This method delegates to the internal `QueryTree` to select informative
     * queries (for exploration/exploitation) used by the GEK mapping.
     */
    std::vector<int> get_next_query() override;

    /**
     * @brief Record an observed value for a coordinate.
     *
     * Observations are stored and passed to the underlying `GEKMapping` for
     * training/updating as needed.
     *
     * @param query coordinate sampled
     * @param result observed value in [0,1]
     */
    void update_prediction(const std::vector<int> &query, double result) override;

    /**
     * @brief Predict the value at a coordinate using the GEK mapping.
     *
     * @param query coordinate to estimate
     * @return double predicted value (clamped to [0,1] by callers)
     */
    double get_value_at(const std::vector<int> &query) override;
    
    /**
     * @brief Configure which dimensions have periodic boundary conditions.
     *
     * This toggles periodic handling inside `GEKMapping` so distance and
     * covariance computations wrap correctly on specified axes.
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