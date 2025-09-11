#if defined(RBF) || defined(TESTING)
#include "RBFModel.hpp"
#include <cmath>
#include <limits>
#include <functional>

RBFModel::RBFModel(int dimensions, int dimensionSize, int totalQueries)
    : Model(dimensions, dimensionSize, totalQueries),
      stateSpace(new ArrayStateSpace(dimensions, dimensionSize)),
      m_sigma(1.0) {}

std::vector<std::vector<int>> RBFModel::generate_all_points() const {
    std::vector<std::vector<int>> points;
    std::vector<int> current(stateSpace->get_dimensions(), 0);

    std::function<void(int)> dfs = [&](int dim) {
        if (dim == stateSpace->get_dimensions()) {
            points.push_back(current);
            return;
        }
        for (int i = 0; i < stateSpace->get_dimension_size(); i++) {
            current[dim] = i;
            dfs(dim + 1);
        }
    };

    dfs(0);
    return points;
}

bool RBFModel::is_queried(const std::vector<int>& point) const {
    for (const auto& [q, _] : m_data) {
        if (q == point) return true;
    }
    return false;
}

std::vector<int> RBFModel::get_next_query() {
    currentQuery++;

    auto candidates = generate_all_points();

    // If no queries yet, pick the first point
    if (m_data.empty()) {
        return candidates.front();
    }

    std::vector<int> bestPoint;
    double bestScore = -1.0;

    for (const auto& point : candidates) {
        if (is_queried(point)) continue;

        double minDist2 = std::numeric_limits<double>::infinity();
        for (const auto& [q, _] : m_data) {
            double dist2 = 0.0;
            for (size_t i = 0; i < point.size(); i++) {
                double d = static_cast<double>(point[i]) - static_cast<double>(q[i]);
                dist2 += d * d;
            }
            if (dist2 < minDist2) minDist2 = dist2;
        }

        if (minDist2 > bestScore) {
            bestScore = minDist2;
            bestPoint = point;
        }
    }

    return bestPoint;
}

void RBFModel::update_prediction(const std::vector<int> &query, double result) {
    m_data.emplace_back(query, result);
    stateSpace->set(query, result);
    if (currentQuery == totalQueries)
        update_prediction_final();
}

double RBFModel::predict(const std::vector<int> &point) const {
    if (m_data.empty()) return 0.0;

    double num = 0.0;
    double den = 0.0;

    for (const auto& [q, val] : m_data) {
        double dist2 = 0.0;
        for (size_t i = 0; i < point.size(); i++) {
            double d = static_cast<double>(point[i]) - static_cast<double>(q[i]);
            dist2 += d * d;
        }
        double w = std::exp(-dist2 / (2 * m_sigma * m_sigma));
        num += w * val;
        den += w;
    }

    return (den > 0.0 ? num / den : 0.0);
}

void RBFModel::update_prediction_final() {
    auto all_points = generate_all_points();
    for (const auto& p : all_points) {
        if (!is_queried(p)) {
            double val = predict(p);
            stateSpace->set(p, val);
        }
    }
}

double RBFModel::get_value_at(const std::vector<int> &query) {
    return this->stateSpace->get(query);
}

#endif // RBF || TESTING
