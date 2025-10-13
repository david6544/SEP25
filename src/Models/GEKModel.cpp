
#if defined(GEK) || defined(TESTING)
#include "GEKModel.hpp"
#include <memory>
#include <random>
#include <cmath>
#include <Eigen/Dense>
#include <algorithm>
#include <limits>
#include <utility>

GEKModel::GEKModel(int dimensions, int dimensionSize, int totalQueries) :
    Model(dimensions, dimensionSize, totalQueries),
    stateSpace(std::make_unique<ArrayStateSpace>(dimensions, dimensionSize)),
    gen(std::random_device{}())
{
    currentQuery = 0;
}

// select indices of the k nearest observed points (squared Euclidean distance)
static std::vector<int> select_k_nearest_indices(const std::vector<std::vector<int>>& observed, const std::vector<int>& query, int k) {
    int n = (int)observed.size();
    if (n == 0) return {};
    k = std::min(k, n);
    std::vector<std::pair<double,int>> dists;
    dists.reserve(n);
    for (int i = 0; i < n; ++i) {
        double d = 0.0;
        const auto &pt = observed[i];
        for (size_t j = 0; j < query.size(); ++j) {
            double diff = double(pt[j]) - double(query[j]);
            d += diff * diff;
        }
        dists.emplace_back(d,i);
    }
    std::nth_element(dists.begin(), dists.begin() + k - 1, dists.end(), [](const auto &a, const auto &b){ return a.first < b.first; });
    std::vector<int> out; out.reserve(k);
    for (int i = 0; i < k; ++i) out.push_back(dists[i].second);
    return out;
}

// build local covariance matrix (m x m) and target vector (m)
static void build_local_cov_and_targets(const std::vector<std::vector<int>>& observed,
                                       const std::vector<double>& values,
                                       const std::vector<int>& indices,
                                       double theta,
                                       Eigen::MatrixXd &C_loc,
                                       Eigen::VectorXd &y_loc) {
    int m = (int)indices.size();
    C_loc = Eigen::MatrixXd(m,m);
    y_loc = Eigen::VectorXd(m);
    for (int i = 0; i < m; ++i) {
        y_loc(i) = values[indices[i]];
        for (int j = 0; j < m; ++j) {
            double dist = 0.0;
            const auto &xi = observed[indices[i]];
            const auto &xj = observed[indices[j]];
            for (size_t d = 0; d < xi.size(); ++d) {
                double diff = double(xi[d]) - double(xj[d]);
                dist += diff * diff;
            }
            C_loc(i,j) = std::exp(-theta * dist);
        }
    }
    for (int i = 0; i < m; ++i) C_loc(i,i) += 1e-6; // small jitter
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
    // If we've already observed every possible point in the discrete grid,
    // avoid spinning forever trying to find a new unique point. In that case
    // return a random point (allow repeats) since no unseen point exists.
    int64_t totalArea = 1;
    for (int d = 0; d < dimensions; ++d) totalArea *= int64_t(dimensionSize);
    if ((int64_t)observed_queries.size() >= totalArea) {
        for (int i = 0; i < dimensions; i++) query[i] = dist(gen);
        return query; // all points seen; return a random (possibly repeated) query
    }

    // Try a bounded number of random attempts to find an unseen point. This keeps
    // average-case fast while avoiding pathological loops.
    const int MAX_RANDOM_TRIES = 1000;
    for (int attempt = 0; attempt < MAX_RANDOM_TRIES; ++attempt) {
        for (int i = 0; i < dimensions; i++) query[i] = dist(gen);
        bool unique = true;
        for (const auto& q : observed_queries) {
            if (q == query) { unique = false; break; }
        }
        if (unique) return query;
    }

    // Fallback: deterministically scan the space and return the first unobserved point.
    // This is O(totalArea * observed) in the worst case, but it only runs when the
    // random attempts fail (rare) and prevents infinite loops.
    std::vector<int> candidate(dimensions);
    // lexicographic enumeration
    std::function<bool(int)> next = [&](int pos) -> bool {
        if (pos == dimensions) {
            // check if candidate is observed
            bool seen = false;
            for (const auto &q : observed_queries) {
                if (q == candidate) { seen = true; break; }
            }
            if (!seen) { query = candidate; return true; }
            return false;
        }
        for (int v = 0; v < dimensionSize; ++v) {
            candidate[pos] = v;
            if (next(pos + 1)) return true;
        }
        return false;
    };
    bool found = next(0);
    if (found) return query;

    // As a last resort (shouldn't happen), return a random point.
    for (int i = 0; i < dimensions; i++) query[i] = dist(gen);
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
    int n = (int)observed_queries.size();
    // If configured to use a local neighbourhood and we have more observations than the neighborhood size,
    // select the k nearest observed points and perform a small GP solve.
    if (use_local_neighborhood && n > local_k) {
        std::vector<int> idx = select_k_nearest_indices(observed_queries, query, local_k);
        Eigen::MatrixXd C_loc;
        Eigen::VectorXd y_loc;
        build_local_cov_and_targets(observed_queries, observed_values, idx, theta, C_loc, y_loc);
        int m = (int)idx.size();
        Eigen::VectorXd c_loc(m);
        for (int i = 0; i < m; ++i) c_loc(i) = covariance(query, observed_queries[idx[i]]);
        Eigen::LDLT<Eigen::MatrixXd> ldlt;
        ldlt.compute(C_loc);
        Eigen::VectorXd alpha;
        if (ldlt.info() == Eigen::Success) alpha = ldlt.solve(y_loc);
        else alpha = C_loc.inverse() * y_loc;
        double mean = c_loc.dot(alpha);
        return mean;
    }

    // Fallback: use full covariance inverse
    Eigen::VectorXd c(n);
    for (int i = 0; i < n; ++i) c(i) = covariance(query, observed_queries[i]);
    Eigen::VectorXd y(n);
    for (int i = 0; i < n; ++i) y(i) = observed_values[i];
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
    if (n == 0) {
        trained = false;
        return;
    }
    // If using local neighbourhood, we avoid building the full n x n inverse for scalability.
    if (use_local_neighborhood && n > local_k) {
        // Defer to local-solves at prediction time. We still mark the model as trained so get_value_at
        // and get_variance can run local solves.
        trained = true;
        return;
    }

    Eigen::MatrixXd C(n, n);
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            C(i, j) = covariance(observed_queries[i], observed_queries[j]);
        }
    }
    // Add regularization to prevent singular matrix
    for(int i = 0; i < n; i++) {
        C(i, i) += 1e-6;
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
    // If using local neighbourhood, compute variance using local k-nearest observed points
    if (use_local_neighborhood && n > local_k) {
        std::vector<int> idx = select_k_nearest_indices(observed_queries, query, local_k);
        Eigen::MatrixXd C_loc;
        Eigen::VectorXd y_loc;
        build_local_cov_and_targets(observed_queries, observed_values, idx, theta, C_loc, y_loc);
        int m = (int)idx.size();
        Eigen::VectorXd c_loc(m);
        for (int i = 0; i < m; ++i) c_loc(i) = covariance(query, observed_queries[idx[i]]);
        Eigen::LDLT<Eigen::MatrixXd> ldlt;
        ldlt.compute(C_loc);
        Eigen::VectorXd v;
        if (ldlt.info() == Eigen::Success) v = ldlt.solve(c_loc);
        else v = C_loc.inverse() * c_loc;
        double var = covariance(query, query) - c_loc.dot(v);
        if (var < 0) var = 0.0;
        return var;
    }

    double var = covariance(query, query) - c.dot(C_inv * c);
    return var;
}

#endif // GEK || TESTING