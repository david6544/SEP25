#if defined(GEK) || defined(TESTING)
#include "GEKModel.hpp"
#include <iostream>
#include <stdexcept>

GEKModel::GEKModel(int dimensions, int dimensionSize, int totalQueries) :
    Model(dimensions, dimensionSize, totalQueries),
    queryTree(nullptr),
    mapping(nullptr),
    leafSize(15),
    use_local_neighborhood(true),
    local_k(64),
    periodic_dims(dimensions, false)
{
    currentQuery = 0;
    queryTree = new QueryTree(dimensions, dimensionSize, leafSize);
}

GEKModel::~GEKModel() {
    if (queryTree) delete queryTree;
    if (mapping) delete mapping;
}

std::vector<int> GEKModel::get_next_query() {
    currentQuery++;
    return queryTree->get_next_query();
}

void GEKModel::update_prediction(const std::vector<int> &query, double result) {
    queryTree->update_prediction(query, result);
    data.emplace_back(result, query);
    
    // Create/recreate the mapping when we've collected all queries
    if (currentQuery == totalQueries) {
        if (mapping) delete mapping;
        mapping = new GEKMapping(data, dimensions, dimensionSize, use_local_neighborhood, local_k);
        if (!periodic_dims.empty()) {
            mapping->set_periodic_dimensions(periodic_dims);
        }
    }
}

double GEKModel::get_value_at(const std::vector<int>& query) {
    if (mapping) {
        return mapping->predict(query);
    }
    // If mapping hasn't been trained yet, return 0
    return 0.0;
}

void GEKModel::set_periodic_dimensions(const std::vector<bool>& periodic) {
    if ((int)periodic.size() != dimensions) {
        throw std::invalid_argument("periodic dimensions vector size mismatch");
    }
    periodic_dims = periodic;
    if (mapping) {
        mapping->set_periodic_dimensions(periodic);
    }
}

#endif // GEK || TESTING
