#include "StochasticQueryModel.hpp"
#include "../InputOutput/InputOutput.hpp"



StochasticQueryModel::StochasticQueryModel(int dimensions, int dimensionSize, int totalQueries) 
    : Model(dimensions, dimensionSize, totalQueries) {
    
    shouldQuery = 10;
    scaleRatio = pow(((double) totalQueries)/(shouldQuery*pow(dimensionSize, dimensions)), 1.0/dimensions);
    invScale = 1.0/scaleRatio;
    scaledSize = floor(dimensionSize * scaleRatio);
    shouldQuery = 4 * shouldQuery;
    qt = new QueryTree(dimensions, scaledSize, 10);
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

    if (tCount >= shouldQuery || currentQuery == totalQueries || ((tCount > 0.75 * shouldQuery) && (wCount >= 0.9 * tCount || wCount <= 0.1 * tCount))) {
        queryWinTotal[currentQueryIDX] = {wCount, tCount};
        double currentProb = double(wCount) / tCount;
        qt->update_prediction(currentQueryPoint, currentProb);
        wCount = tCount = repeatQueries = 0;
    }

    // When we've processed all overall queries, build the interpolator
    if (currentQuery >= totalQueries) {
        for (int i = 0; i < queryWinTotal.size(); i++) {
            if (queryWinTotal[i].empty()) continue;
            double prob = double(queryWinTotal[i][0]) / queryWinTotal[i][1];
            std::vector<int> cQ = unscale_midpoint(InputOutput::index_to_coords(i, dimensions, scaledSize));
            data.emplace_back(prob, cQ);
        }
        maper = new LinearStochastic(data, 1 << (int)query.size(), 1, 0.1);
    }
}

double StochasticQueryModel::get_value_at(const std::vector<int>& query) {
    return maper->predict(query);
}