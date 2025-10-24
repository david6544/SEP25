#include "StochasticQueryModel.hpp"



StochasticQueryModel::StochasticQueryModel(int dimensions, int dimensionSize, int totalQueries) 
    : Model(dimensions, dimensionSize, totalQueries) {
    
    shouldQuery = 5;
    scaleRatio = pow(((double) totalQueries)/(shouldQuery*pow(dimensionSize, dimensions)), 1.0/dimensions);
    scaledSize = floor(dimensionSize * scaleRatio);

    qt = new QueryTree(dimensions, scaledSize, 10);
    totalPoints = pow(scaledSize, dimensions);
    
}

StochasticQueryModel::~StochasticQueryModel(){
    if (this->maper) delete this->maper;
    if (this->qt) delete this->qt;
}

std::vector<int> StochasticQueryModel::lowerResolution(const std::vector<int>& rawQuery) {
    std::vector<int> scaled(rawQuery.size());
    for (size_t i = 0; i < rawQuery.size(); ++i) {
        scaled[i] = std::min(scaledSize - 1, (int) std::floor(rawQuery[i] * scaleRatio));
    }
    return scaled;
}

std::vector<int> StochasticQueryModel::unscaleCandidate(const std::vector<int>& scaledQuery) {
    std::vector<int> unscaled(scaledQuery.size());
    for (size_t i = 0; i < scaledQuery.size(); ++i) {
        // pick the middle of the interval deterministically
        double lower = scaledQuery[i] / scaleRatio;
        double upper = (scaledQuery[i] + 1) / scaleRatio;
        unscaled[i] = (int) std::floor((lower + upper) / 2.0);
    }
    return unscaled;
}


std::vector<int> StochasticQueryModel::get_next_query() {
    currentQuery++;

    if (repeatQueries == 0) {
        // new point chosen
        currentQueryPoint = qt->get_next_query();
        repeatQueries = shouldQuery;
        // reset per-point counters
        wCount = 0;
        tCount = 0;
    } else {
        // still repeating same point
        repeatQueries--;
    }

    return unscaleCandidate(currentQueryPoint);
}

void StochasticQueryModel::update_prediction(const std::vector<int>& query, double result) {
    // sanity: if query doesn't match currentQueryPoint, optionally handle it
    // (here we expect they match)
    std::vector<int> scaledQuery = lowerResolution(query);
    if (scaledQuery != currentQueryPoint) {
        std::cerr << "Warning: update_prediction called with non-current query\n";
    }

    // increment per-point counters
    if (result >= 0.9) wCount++;
    tCount++;

    bool shouldSkip = (tCount > 0.75 * shouldQuery) && (wCount >= 0.9 * tCount || wCount <= 0.1 * tCount);

    // When we've collected enough samples for this point, flush them to the map
    if (tCount >= shouldQuery || shouldSkip) {
        auto &currentResults = queryWinTotal[scaledQuery]; // creates if missing
        if (currentResults.size() < 2) currentResults.assign(2, 0);
        // accumulate
        currentResults[0] += wCount; // wins
        currentResults[1] += tCount; // trials

        double currentProb = double(currentResults[0]) / currentResults[1];
        // update the tree with the aggregated probability
        qt->update_prediction(query, currentProb);

        // reset per-point counters so next time we start fresh for new selection
        wCount = 0;
        tCount = 0;
        repeatQueries = 0;
    }

    // When we've processed all overall queries, build the interpolator
    if (currentQuery >= totalQueries) {
        data.clear();
        for (auto &p : queryWinTotal) {
            if (p.second[1] == 0) continue;
            double prob = double(p.second[0]) / p.second[1];
            data.emplace_back(prob, p.first);
        }
        if (maper) delete maper;
        maper = new IDW(data, 4, 2);
    }
}

double StochasticQueryModel::get_value_at(const std::vector<int>& query) {
    if (maper) return maper->predict(lowerResolution(query));
    return 0.0;
}