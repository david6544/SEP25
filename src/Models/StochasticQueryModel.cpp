#include "StochasticQueryModel.hpp"
#include "../InputOutput/InputOutput.hpp"



StochasticQueryModel::StochasticQueryModel(int dimensions, int dimensionSize, int totalQueries) 
    : Model(dimensions, dimensionSize, totalQueries) {
    
    shouldQuery = 50;
    scaleRatio = pow(((double) totalQueries)/(shouldQuery*pow(dimensionSize, dimensions)), 1.0/dimensions);
    scaledSize = floor(dimensionSize * scaleRatio);
    shouldQuery = 4 * shouldQuery;
    qt = new QueryTree(dimensions, scaledSize, 10);
    totalPoints = pow(scaledSize, dimensions);
    
    precomputedBounds.resize(scaledSize);
    const double invScale = 1.0 / scaleRatio;

    for (int s = 0; s < scaledSize; ++s) {
        int lo = static_cast<int>(std::ceil(s * invScale));
        int hi = static_cast<int>(std::ceil((s + 1) * invScale)) - 1;
        lo = std::clamp(lo, 0, dimensionSize - 1);
        hi = std::clamp(hi, 0, dimensionSize - 1);
        precomputedBounds[s] = { lo, hi };
    }
    precomputedBounds[0].first = 0;
    precomputedBounds[scaledSize-1].second = dimensionSize;
}

StochasticQueryModel::~StochasticQueryModel(){
    if (this->maper) delete this->maper;
    if (this->qt) delete this->qt;
}

std::vector<int> StochasticQueryModel::lowerResolution(const std::vector<int>& rawQuery) const {
    std::vector<int> scaled(rawQuery.size());
    double invScale = 1.0 / scaleRatio;

    for (size_t i = 0; i < rawQuery.size(); ++i) {
        int x = rawQuery[i];
        // Direct scaled estimate
        int s = static_cast<int>(x * scaleRatio);

        // Correct potential boundary rounding using integer math
        double lowerBound = s * invScale;
        double upperBound = (s + 1) * invScale;
        if (x < lowerBound) s--;
        else if (x >= upperBound) s++;

        scaled[i] = std::clamp(s, 0, scaledSize - 1);
    }
    return scaled;
}

std::vector<int> StochasticQueryModel::sample_unscaled_point_from_scaled(const std::vector<int>& scaledQuery) const{
    std::vector<int> result(scaledQuery.size());
    double invScale = 1.0 / scaleRatio;
    static thread_local std::mt19937_64 rng(std::random_device{}());

    for (size_t i = 0; i < scaledQuery.size(); ++i) {
        int s = scaledQuery[i];

        int lo = std::clamp(precomputedBounds[s].first, 0, dimensionSize - 1);
        int hi = std::clamp(precomputedBounds[s].second, 0, dimensionSize - 1);

        if (lo < hi) {
            std::uniform_int_distribution<int> dist(lo, hi);
            result[i] = dist(rng);
        } else {
            result[i] = lo; // degenerate case (single value)
        }
    }
    return result;
}


std::vector<int> StochasticQueryModel::unscale_midpoint(const std::vector<int>& scaledQuery) const {
    std::vector<int> unscaled(scaledQuery.size());
    double invScale = 1.0 / scaleRatio;

    for (size_t i = 0; i < scaledQuery.size(); ++i) {
        int s = scaledQuery[i];
        int lo = static_cast<int>(std::ceil(s * invScale));
        int hi = static_cast<int>(std::ceil((s + 1) * invScale)) - 1;
        lo = std::clamp(lo, 0, dimensionSize - 1);
        hi = std::clamp(hi, 0, dimensionSize - 1);
        unscaled[i] = (lo + hi) / 2;
    }
    return unscaled;
}

inline int linear_index(const std::vector<int>& coords, int scaledSize) {
    int idx = 0;
    for (size_t d = 0; d < coords.size(); ++d) {
        idx = idx * scaledSize + coords[d];
    }
    return idx;
}

std::vector<int> StochasticQueryModel::get_next_query() {
    currentQuery++;

    if (repeatQueries == 0) {
        // new point chosen
        currentQueryPoint = qt->get_next_query();
        currentQueryIDX = linear_index(currentQueryPoint, scaledSize);
        repeatQueries = shouldQuery;
        // reset per-point counters
        wCount = 0;
        tCount = 0;
    } else {
        // still repeating same point
        repeatQueries--;
    }

    return sample_unscaled_point_from_scaled(currentQueryPoint);
}


void StochasticQueryModel::update_prediction(const std::vector<int>& query, double result) {
    // increment per-point counters
    if (result >= 0.9) wCount++;
    tCount++;

    bool shouldSkip = (tCount > 0.75 * shouldQuery) && (wCount >= 0.9 * tCount || wCount <= 0.1 * tCount);

    // When we've collected enough samples for this point, flush them to the map
    if (tCount >= shouldQuery || shouldSkip || currentQuery == totalQueries) {
        auto &currentResults = queryWinTotal[currentQueryIDX]; // creates if missing
        if (currentResults.size() < 2) currentResults.assign(2, 0);
        // accumulate
        currentResults[0] += wCount; // wins
        currentResults[1] += tCount; // trials

        double currentProb = double(currentResults[0]) / currentResults[1];
        // update the tree with the aggregated probability
        qt->update_prediction(currentQueryPoint, currentProb);

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
            data.emplace_back(prob, unscale_midpoint(InputOutput::index_to_coords(p.first, dimensions, scaledSize)));
        }
        if (maper) delete maper;
        double neigh = std::max(5.0, 0.005 * totalPoints);
        maper = new IDW(data, neigh, 1, 0.1);
    }
}

double StochasticQueryModel::get_value_at(const std::vector<int>& query) {
    if (maper) return maper->predict(query);
    return 0.0;
}