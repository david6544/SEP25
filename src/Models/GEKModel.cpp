
#if defined(GEK) || defined(TESTING)
#include "GEKModel.hpp"
#include <memory>
#include <random>
#include <cmath>
#include <Eigen/Dense>

GEKModel::GEKModel(int dimensions, int dimensionSize, int totalQueries) :
    Model(dimensions, dimensionSize, totalQueries),
    stateSpace(std::make_unique<ArrayStateSpace>(dimensions, dimensionSize)),
    gen(std::random_device{}())
{
    currentQuery = 0;
}

GEKModel::~GEKModel() {
    // unique_ptr will handle
}

std::vector<int> GEKModel::get_next_query() {
    currentQuery++;
    if (!trained || dimensions > 2) {
        return get_random_query();
    } else {
        // find query with max variance
        double max_var = -1.0;
        std::vector<int> best_query;
        for(int i = 0; i < dimensionSize; i++) {
            for(int j = 0; j < dimensionSize; j++) {
                std::vector<int> q = {i, j};
                double v = get_variance(q);
                if(v > max_var) {
                    max_var = v;
                    best_query = q;
                }
            }
        }
        return best_query;
    }
}

std::vector<int> GEKModel::get_random_query() {
    std::vector<int> query(dimensions);
    std::uniform_int_distribution<> dist(0, dimensionSize - 1);
    bool unique = false;
    while (!unique) {
        for (int i = 0; i < dimensions; i++) {
            query[i] = dist(gen);
        }
        unique = true;
        for (const auto& q : observed_queries) {
            if (q == query) {
                unique = false;
                break;
            }
        }
    }
    return query;
}

void GEKModel::update_prediction(const std::vector<int> &query, double result) {
    observed_queries.push_back(query);
    observed_values.push_back(result);
    stateSpace->set(query, result);
    if (currentQuery == totalQueries) {
        train();
    }
}

double GEKModel::get_value_at(const std::vector<int>& query) {
    if (!trained) {
        return 0.0;
    }
    int n = observed_queries.size();
    Eigen::VectorXd c(n);
    for(int i = 0; i < n; i++) {
        c(i) = covariance(query, observed_queries[i]);
    }
    Eigen::VectorXd y(n);
    for(int i = 0; i < n; i++) {
        y(i) = observed_values[i];
    }
    double mean = c.dot(C_inv * y);
    return mean;
}

double GEKModel::covariance(const std::vector<int>& x, const std::vector<int>& y) {
    double dist = 0.0;
    for(size_t i = 0; i < x.size(); i++) {
        double d = x[i] - y[i];
        dist += d * d;
    }
    return exp(-theta * dist);
}

void GEKModel::train() {
    int n = observed_queries.size();
    Eigen::MatrixXd C(n, n);
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            C(i, j) = covariance(observed_queries[i], observed_queries[j]);
        }
    }
    // Add regularization to prevent singular matrix
    for(int i = 0; i < n; i++) {
        C(i, i) += 1e-4;
    }
    C_inv = C.inverse();
    trained = true;
}

double GEKModel::get_variance(const std::vector<int>& query) {
    if (!trained) return 1.0;
    int n = observed_queries.size();
    Eigen::VectorXd c(n);
    for(int i = 0; i < n; i++) {
        c(i) = covariance(query, observed_queries[i]);
    }
    double var = covariance(query, query) - c.dot(C_inv * c);
    return var;
}

#endif // GEK || TESTING