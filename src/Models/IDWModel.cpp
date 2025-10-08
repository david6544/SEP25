#if defined(IDW) || defined(TESTING)
#include "IDWModel.hpp"


IDWModel::IDWModel(int dimensions, int dimensionSize, int totalQueries) : Model(dimensions, dimensionSize, totalQueries) {
    this->currentQuery = 0;
    this->stateSpace = new KDTreeStateSpace(dimensions, dimensionSize);
}

double IDWModel::predictValueAt(const std::vector<int>& query) {
    // Get nearest nodes or bucket
    auto neighbors = stateSpace->nearest_neighbor(query);
    if (neighbors.empty()) return 0.0;

    double weightedSum = 0.0;
    double weightTotal = 0.0;
    const double epsilon = 1e-9;

    for (auto* node : neighbors) {
        double dist2 = stateSpace->squared_distance(query, node->coords);
        double dist = std::sqrt(dist2 + epsilon);

        // Basic inverse distance weight
        double w = 1.0 / dist;

        // Adjust for exploration (less explored → higher weight)
        double explorationFactor = 1.0 / (1.0 + node->visitCount);

        // Adjust for accuracy (more error → lower confidence)
        double accuracyFactor = 1.0 / (1.0 + node->totalError);

        double combinedWeight = w * explorationFactor * accuracyFactor;

        weightedSum += combinedWeight * node->value;
        weightTotal += combinedWeight;
    }

    return (weightTotal > 0.0) ? weightedSum / weightTotal : 0.0;
}



std::vector<int> IDWModel::get_next_query() {
    auto root = stateSpace->get_root();
    if (!root) {
        // First query: start in the middle
        std::vector<int> mid(dimensions, dimensionSize / 2);
        return mid;
    }

    // Sample a few random points and pick one with the highest uncertainty
    std::vector<int> bestQuery(dimensions);
    double lowestConfidence = std::numeric_limits<double>::max();

    for (int i = 0; i < 10; ++i) {
        std::vector<int> candidate(dimensions);
        for (int d = 0; d < dimensions; ++d) {
            candidate[d] = rand() % dimensionSize;
        }

        double predicted = predictValueAt(candidate);
        auto neighbors = stateSpace->nearest_neighbor(candidate);
        double avgError = 0.0;
        for (auto* n : neighbors) avgError += n->totalError;
        avgError /= std::max(1, (int)neighbors.size());

        double confidence = avgError + (neighbors.empty() ? 1.0 : neighbors.front()->visitCount);
        if (confidence < lowestConfidence) {
            lowestConfidence = confidence;
            bestQuery = candidate;
        }
    }

    return bestQuery;
}



void IDWModel::update_prediction(const std::vector<int>& query, double result) {
    double predicted = predictValueAt(query);
    double error = std::abs(predicted - result);

    // Store the actual value
    stateSpace->insert(query, result);

    // Update local exploration and error stats
    auto neighbors = stateSpace->nearest_neighbor(query);
    for (auto* node : neighbors) {
        node->visitCount += 1;
        node->totalError += error;
    }
}



double IDWModel::get_value_at(const std::vector<int>& query) {
    return stateSpace->get(query);
}

#endif // IDW || TESTING