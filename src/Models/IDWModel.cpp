#if defined(IDW) || defined(TESTING)
#include "IDWModel.hpp"
#include <functional>

// Helper to hash coordinates for visited set
std::string hash_coords(const std::vector<int>& coords) {
    std::string h;
    for (auto v : coords) h += std::to_string(v) + ",";
    return h;
}

IDWModel::IDWModel(int dimensions, int dimensionSize, int totalQueries) : Model(dimensions, dimensionSize, totalQueries) {
    this->currentQuery = 0;
    this->stateSpace = new KDTreeStateSpace(dimensions, dimensionSize);
}

double IDWModel::predictValueAt(const std::vector<int>& query) {
    auto neighbors = stateSpace->nearest_neighbor(query);
    if (neighbors.empty()) return 0.0;

    double weightedSum = 0.0;
    double weightTotal = 0.0;
    const double epsilon = 1e-9;

    std::cout << neighbors.size() << " neighbors found for prediction at ";
    
    for (auto* node : neighbors) {
        double dist2 = stateSpace->squared_distance(query, node->coords);
        double dist = std::sqrt(dist2 + epsilon);
        double w = 1.0 / dist; // basic inverse distance

        // Exploration factor (less visited → higher weight)
        double explorationFactor = 1.0 / (1.0 + node->visitCount);

        // Accuracy factor (high error → lower weight)
        double accuracyFactor = 1.0 / (1.0 + node->totalError);

        // Uncertainty factor (high variance → lower weight)
        double variance = (node->visitCount > 1) ? node->M2 / (node->visitCount - 1) : 0.0;
        double uncertaintyFactor = 1.0 / (1.0 + variance);

        double combinedWeight = w * explorationFactor * accuracyFactor * uncertaintyFactor;

        weightedSum += combinedWeight * node->value;
        weightTotal += combinedWeight;
    }

    return (weightTotal > 0.0) ? weightedSum / weightTotal : 0.0;
}




std::vector<int> IDWModel::get_next_query() {
    ++currentQuery;

    // --- Step 1: Final query — fill the entire state space ---
    if (currentQuery >= totalQueries) {
        std::vector<int> coords(dimensions, 0);

        // Recursive helper to enumerate all points in N-dimensional space
        std::function<void(int)> fillSpace;
        fillSpace = [&](int dim) {
            if (dim == dimensions) {
                // Predict and insert
                double predicted = predictValueAt(coords);
                stateSpace->insert(coords, predicted);
                visited.insert(hash_coords(coords));
                return;
            }
            for (int i = 0; i < dimensionSize; ++i) {
                coords[dim] = i;
                fillSpace(dim + 1);
            }
        };

        fillSpace(0);

        // Return a dummy query (won't matter)
        return std::vector<int>(dimensions, 0);
    }

    // --- Step 2: Select next query normally ---
    auto root = stateSpace->get_root();
    if (!root) {
        std::vector<int> mid(dimensions, dimensionSize / 2);
        visited.insert(hash_coords(mid));
        return mid;
    }

    std::vector<int> bestQuery(dimensions);
    double lowestConfidence = std::numeric_limits<double>::max();
    bool found = false;

    for (int i = 0; i < 100; ++i) { // Increased trials for better coverage
        std::vector<int> candidate(dimensions);
        for (int d = 0; d < dimensions; ++d) {
            candidate[d] = rand() % dimensionSize;
        }

        if (visited.count(hash_coords(candidate)) > 0) continue;

        auto neighbors = stateSpace->nearest_neighbor(candidate);
        double avgError = 0.0;
        for (auto* n : neighbors) avgError += n->totalError;
        avgError /= std::max(1, (int)neighbors.size());

        double confidence = avgError + (neighbors.empty() ? 1.0 : neighbors.front()->visitCount);

        if (confidence < lowestConfidence) {
            lowestConfidence = confidence;
            bestQuery = candidate;
            found = true;
        }
    }

    // --- Step 3: Fallback if random sampling didn't find an unvisited point ---
    if (!found) {
        std::vector<int> coords(dimensions, 0);
        std::function<bool(int)> findUnvisited;
        findUnvisited = [&](int dim) -> bool {
            if (dim == dimensions) {
                if (visited.count(hash_coords(coords)) == 0) {
                    bestQuery = coords;
                    return true;
                }
                return false;
            }
            for (int i = 0; i < dimensionSize; ++i) {
                coords[dim] = i;
                if (findUnvisited(dim + 1)) return true;
            }
            return false;
        };
        findUnvisited(0);
    }

    visited.insert(hash_coords(bestQuery));
    return bestQuery;
}





void IDWModel::update_prediction(const std::vector<int>& query, double result) {
    double predicted = predictValueAt(query);
    double error = std::abs(predicted - result);

    // Insert or update the actual value
    stateSpace->insert(query, result);

    // Update neighbors with variance info
    auto neighbors = stateSpace->nearest_neighbor(query);
    for (auto* node : neighbors) {
        node->visitCount += 1;

        // Welford's online algorithm for variance
        double delta = result - node->meanValue;
        node->meanValue += delta / node->visitCount;
        double delta2 = result - node->meanValue;
        node->M2 += delta * delta2;

        node->totalError += error;
    }
}




double IDWModel::get_value_at(const std::vector<int>& query) {
    return stateSpace->get(query);
}

#endif // IDW || TESTING