#if defined(GEK) || defined(TESTING)
#include "GEKMapping.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

// Helper function to select indices of the k nearest observed points (squared Euclidean distance)
static std::vector<int> select_k_nearest_indices_helper(
    const std::vector<std::pair<double, std::vector<int>>>& observed,
    const std::vector<int>& query,
    int k)
{
    int n = (int)observed.size();
    if (n == 0) return {};
    k = std::min(k, n);
    
    std::vector<std::pair<double, int>> dists;
    dists.reserve(n);
    
    for (int i = 0; i < n; ++i) {
        const auto& coords = observed[i].second;
        double d2 = 0.0;
        for (size_t j = 0; j < coords.size(); ++j) {
            double diff = coords[j] - query[j];
            d2 += diff * diff;
        }
        dists.push_back({d2, i});
    }
    
    std::nth_element(dists.begin(), dists.begin() + k - 1, dists.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });
    
    std::vector<int> out;
    out.reserve(k);
    for (int i = 0; i < k; ++i) {
        out.push_back(dists[i].second);
    }
    return out;
}

GEKMapping::GEKMapping(std::vector<std::pair<double, std::vector<int>>>& data,
                       int dimensions,
                       int dimensionSize,
                       bool use_local_neighborhood,
                       int local_k)
    : Mapping(data),
      dimensions(dimensions),
      dimensionSize(dimensionSize),
      use_local_neighborhood(use_local_neighborhood),
      local_k(local_k),
      periodic_dims(dimensions, false)
{
    // If not using local neighborhoods, train global model immediately
    if (!use_local_neighborhood) {
        train();
    }
}

GEKMapping::~GEKMapping() {
    // Eigen matrices will clean up automatically
}

double GEKMapping::covariance(const std::vector<int>& x, const std::vector<int>& y) {
    double d2 = compute_periodic_distance_squared(x, y);
    return std::exp(-theta * d2);
}

double GEKMapping::compute_periodic_distance_squared(const std::vector<int>& x, const std::vector<int>& y) {
    double d2 = 0.0;
    for (int d = 0; d < dimensions; ++d) {
        double diff = x[d] - y[d];
        if (periodic_dims[d]) {
            // For periodic dimensions, use minimum distance accounting for wraparound
            double wrap_diff = dimensionSize - std::abs(diff);
            diff = (std::abs(diff) < wrap_diff) ? diff : (diff > 0 ? -wrap_diff : wrap_diff);
        }
        d2 += diff * diff;
    }
    return d2;
}

void GEKMapping::set_periodic_dimensions(const std::vector<bool>& periodic) {
    if ((int)periodic.size() != dimensions) {
        throw std::invalid_argument("periodic dimensions vector size mismatch");
    }
    periodic_dims = periodic;
    // If global model was trained, need to retrain with new distance metric
    if (trained) {
        train();
    }
}

void GEKMapping::train() {
    int n = (int)queriedPoints.size();
    if (n == 0) return;
    
    C_inv = Eigen::MatrixXd(n, n);
    Eigen::MatrixXd C(n, n);
    
    // Build covariance matrix
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C(i, j) = covariance(queriedPoints[i].second, queriedPoints[j].second);
        }
        // Add small jitter for numerical stability
        C(i, i) += 1e-8;
    }
    
    // Invert using Eigen
    C_inv = C.inverse();
    trained = true;
}

std::vector<int> GEKMapping::select_k_nearest_indices(const std::vector<int>& query, int k) {
    return select_k_nearest_indices_helper(queriedPoints, query, k);
}

void GEKMapping::build_local_cov_and_targets(const std::vector<int>& indices,
                                             Eigen::MatrixXd& C_loc,
                                             Eigen::VectorXd& y_loc)
{
    int m = (int)indices.size();
    C_loc = Eigen::MatrixXd(m, m);
    y_loc = Eigen::VectorXd(m);
    
    for (int i = 0; i < m; ++i) {
        int idx_i = indices[i];
        y_loc(i) = queriedPoints[idx_i].first;
        
        for (int j = 0; j < m; ++j) {
            int idx_j = indices[j];
            C_loc(i, j) = covariance(queriedPoints[idx_i].second, queriedPoints[idx_j].second);
        }
        // Add small jitter for numerical stability
        C_loc(i, i) += 1e-8;
    }
}

double GEKMapping::predict(std::vector<int> query) {
    int n = (int)queriedPoints.size();
    if (n == 0) return 0.0;
    
    // Check if query is an exact match with an observed point
    for (const auto& pt : queriedPoints) {
        if (pt.second == query) {
            return pt.first;
        }
    }
    
    // Use local neighborhood if configured and we have enough points
    if (use_local_neighborhood && n > local_k) {
        std::vector<int> indices = select_k_nearest_indices(query, local_k);
        int m = (int)indices.size();
        if (m == 0) return 0.0;
        
        Eigen::MatrixXd C_loc;
        Eigen::VectorXd y_loc;
        build_local_cov_and_targets(indices, C_loc, y_loc);
        
        // Build covariance vector between query and local points
        Eigen::VectorXd k_vec(m);
        for (int i = 0; i < m; ++i) {
            int idx = indices[i];
            k_vec(i) = covariance(query, queriedPoints[idx].second);
        }
        
        // Solve local system: C_loc * alpha = y_loc
        Eigen::VectorXd alpha = C_loc.colPivHouseholderQr().solve(y_loc);
        
        // Prediction: k_vec^T * alpha
        return k_vec.dot(alpha);
    }
    
    // Global prediction
    if (!trained) {
        train();
    }
    
    if (!trained || n == 0) return 0.0;
    
    // Build covariance vector
    Eigen::VectorXd k_vec(n);
    for (int i = 0; i < n; ++i) {
        k_vec(i) = covariance(query, queriedPoints[i].second);
    }
    
    // Build value vector
    Eigen::VectorXd y(n);
    for (int i = 0; i < n; ++i) {
        y(i) = queriedPoints[i].first;
    }
    
    // Prediction: k_vec^T * C_inv * y
    return k_vec.dot(C_inv * y);
}

double GEKMapping::get_variance(const std::vector<int>& query) {
    int n = (int)queriedPoints.size();
    if (n == 0) return 1.0;  // Maximum uncertainty when no data
    
    // Use local neighborhood if configured
    if (use_local_neighborhood && n > local_k) {
        std::vector<int> indices = select_k_nearest_indices(query, local_k);
        int m = (int)indices.size();
        if (m == 0) return 1.0;
        
        Eigen::MatrixXd C_loc;
        Eigen::VectorXd y_loc;
        build_local_cov_and_targets(indices, C_loc, y_loc);
        
        Eigen::VectorXd k_vec(m);
        for (int i = 0; i < m; ++i) {
            int idx = indices[i];
            k_vec(i) = covariance(query, queriedPoints[idx].second);
        }
        
        Eigen::VectorXd v = C_loc.colPivHouseholderQr().solve(k_vec);
        double var = 1.0 - k_vec.dot(v);
        return std::max(0.0, var);
    }
    
    // Global variance
    if (!trained) {
        const_cast<GEKMapping*>(this)->train();
    }
    
    if (!trained) return 1.0;
    
    Eigen::VectorXd k_vec(n);
    for (int i = 0; i < n; ++i) {
        k_vec(i) = covariance(query, queriedPoints[i].second);
    }
    
    double var = 1.0 - k_vec.dot(C_inv * k_vec);
    return std::max(0.0, var);
}

#endif // GEK || TESTING
