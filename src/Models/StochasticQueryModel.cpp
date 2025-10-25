#include "StochasticQueryModel.hpp"
#include "../InputOutput/InputOutput.hpp"



StochasticQueryModel::StochasticQueryModel(int dimensions, int dimensionSize, int totalQueries) 
    : Model(dimensions, dimensionSize, totalQueries) {
    
    // Adaptive sampling strategy for complex spaces
    // Reduced base samples for faster runtime, will stop early when confident
    shouldQuery = (dimensions >= 3) ? 18 : 10;  // Reduced from 22
    
    // Scale ratio calculation with better coverage for complex spaces
    double targetPoints = ((double)totalQueries) / (shouldQuery * 1.35); // Slightly tighter
    scaleRatio = pow(targetPoints / pow(dimensionSize, dimensions), 1.0/dimensions);
    invScale = 1.0/scaleRatio;
    scaledSize = std::max((dimensions >= 3 ? 5 : 3), (int)floor(dimensionSize * scaleRatio));
    
    // More repeats for stochastic estimation accuracy
    shouldQuery = 4 * shouldQuery;  // 72 samples for 3D
    
    // Smaller leaf size for better local adaptation to discontinuities
    int leafSize = (dimensions >= 3) ? 7 : 10;
    qt = new QueryTree(dimensions, scaledSize, leafSize);
    
    totalPoints = pow(scaledSize, dimensions);
    queryWinTotal.resize(totalPoints);
    
    precomputedBounds.resize(scaledSize);

    for (int s = 0; s < scaledSize; ++s) {
        int lo = static_cast<int>(std::ceil(s * invScale));
        int hi = static_cast<int>(std::ceil((s + 1) * invScale)) - 1;
        lo = std::clamp(lo, 0, dimensionSize - 1);
        hi = std::clamp(hi, 0, dimensionSize - 1);
        precomputedBounds[s] = { lo, hi };
    }
    precomputedBounds[0].first = 0;
    precomputedBounds[scaledSize-1].second = dimensionSize-1;
}

StochasticQueryModel::~StochasticQueryModel(){
    if (this->maper) delete this->maper;
    if (this->qt) delete this->qt;
}

std::vector<int> StochasticQueryModel::lowerResolution(const std::vector<int>& rawQuery) const {
    std::vector<int> scaled(rawQuery.size());

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
    static thread_local std::mt19937_64 rng(std::random_device{}());

    for (size_t i = 0; i < scaledQuery.size(); ++i) {
        int s = scaledQuery[i];
        result[i] = precomputedBounds[s].first + rng() % (precomputedBounds[s].second - precomputedBounds[s].first +1);
    }
    return result;
}


std::vector<int> StochasticQueryModel::unscale_midpoint(const std::vector<int>& scaledQuery) const {
    std::vector<int> unscaled(scaledQuery.size());
    for (size_t i = 0; i < scaledQuery.size(); ++i) {
        int s = scaledQuery[i];
        unscaled[i] = (precomputedBounds[s].first + precomputedBounds[s].second) / 2;
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
        wCount = tCount = 0;
    }

    repeatQueries--;

    return sample_unscaled_point_from_scaled(currentQueryPoint);
}


void StochasticQueryModel::update_prediction(const std::vector<int>& query, double result) {
    if (result >= 0.9) wCount++;
    tCount++;

    // More aggressive early stopping for runtime optimization
    // Stop after 60% samples if extremely confident (>95% or <5%)
    bool veryEarlyStop = (tCount > 0.6 * shouldQuery) && 
                         (wCount >= 0.97 * tCount || wCount <= 0.03 * tCount);
    
    // Standard early stop after 70% if confident
    bool earlyStop = (tCount > 0.7 * shouldQuery) && 
                     (wCount >= 0.93 * tCount || wCount <= 0.07 * tCount);
    
    bool shouldUpdate = (tCount >= shouldQuery || currentQuery == totalQueries || veryEarlyStop || earlyStop);
    
    if (shouldUpdate) {
        queryWinTotal[currentQueryIDX] = {wCount, tCount};
        double currentProb = double(wCount) / tCount;
        qt->update_prediction(currentQueryPoint, currentProb);
        wCount = tCount = repeatQueries = 0;
    }

    // When we've processed all overall queries, build the interpolator
    if (currentQuery >= totalQueries) {
        // Pre-allocate for efficiency
        data.reserve(totalPoints / 2);
        
        for (int i = 0; i < queryWinTotal.size(); i++) {
            if (queryWinTotal[i].empty()) continue;
            double prob = double(queryWinTotal[i][0]) / queryWinTotal[i][1];
            data.emplace_back(prob, unscale_midpoint(InputOutput::index_to_coords(i, dimensions, scaledSize)));
        }
        
        // Determine wrap axis: y-axis (index 1) for 3D, x-axis (index 0) for 2D
        int wrapAxis = (dimensions == 3) ? 1 : (dimensions == 2 ? 0 : -1);
        
        // Optimized neighbor count - balance accuracy vs speed
        double densityRatio = (double)data.size() / totalPoints;
        int baseNeigh = 16 + dimensions * 6; // Reduced for speed
        int neigh = std::clamp((int)(baseNeigh * densityRatio * 2), 
                               (dimensions >= 3 ? 10 : 8), 
                               120);  // Reduced max

        
        // Higher power for better discontinuity handling, smaller offset for complex spaces
        int adaptivePower = (dimensions >= 3) ? 2 : 1;
        double adaptiveOffset = (dimensions >= 3) ? 0.03 : 0.05;  // Tighter for 3D
        
        maper = new IDW(data, neigh, adaptivePower, adaptiveOffset, wrapAxis, dimensionSize);
    }
}

double StochasticQueryModel::get_value_at(const std::vector<int>& query) {
    if (!maper) return 0.0; // Return default value if interpolator not yet initialized
    return maper->predict(query);
}