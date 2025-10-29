#include "StochasticQueryModel.hpp"
#include "../InputOutput/InputOutput.hpp"


StochasticQueryModel::StochasticQueryModel(int dimensions, int dimensionSize, int totalQueries) 
    : Model(dimensions, dimensionSize, totalQueries) {
    
    shouldQuery = 200;
    scaleRatio = std::min(1.0, pow(((double) totalQueries)/(shouldQuery*pow(dimensionSize, dimensions)), 1.0/dimensions));
    if (scaleRatio == 1) shouldQuery = totalQueries/(pow(dimensionSize, dimensions));
    invScale = 1/scaleRatio;
    scaledSize = floor(dimensionSize * scaleRatio);
    shouldQuery = 1.5*shouldQuery;
    qt = new QueryTree(dimensions, scaledSize, 30);
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
        int r1 = (precomputedBounds[s].first + rng() % (precomputedBounds[s].second - precomputedBounds[s].first +1))/4;
        int r2 = (precomputedBounds[s].first + rng() % (precomputedBounds[s].second - precomputedBounds[s].first +1))/4;
        int r3 = (precomputedBounds[s].first + rng() % (precomputedBounds[s].second - precomputedBounds[s].first +1))/4;
        int r4 = (precomputedBounds[s].first + rng() % (precomputedBounds[s].second - precomputedBounds[s].first +1))/4;
        result[i] = r1 + r2 + r3 + r4;
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


// Add a prior probability member to your class
double priorProb = 0.315; // initial guess for unknown points
double shrinkFactor = 0.3; // how strongly we shrink toward the prior

void StochasticQueryModel::update_prediction(const std::vector<int>& query, double result) {
    if (result >= 0.9) wCount++;
    tCount++;

    if (tCount >= shouldQuery || (currentQuery == totalQueries && tCount >= 0.75 * shouldQuery)) {
        
        // compute raw probability
        double rawProb = double(wCount) / tCount;

        // shrink toward prior to reduce variance
        double prob = shrinkFactor * rawProb + (1.0 - shrinkFactor) * priorProb;

        queryWinTotal[currentQueryIDX] = { wCount, tCount };
        qt->update_prediction(currentQueryPoint, prob);

        // reset counters
        wCount = tCount = repeatQueries = 0;
    }

    if (currentQuery >= totalQueries) {
        // build interpolator
        data.clear();
        for (int i = 0; i < queryWinTotal.size(); i++) {
            if (queryWinTotal[i].empty()) continue;

            double rawProb = double(queryWinTotal[i][0]) / queryWinTotal[i][1];
            double prob = shrinkFactor * rawProb + (1.0 - shrinkFactor) * priorProb;

            // clamp predictions to avoid extremes
            // prob = std::clamp(prob, 0.05, 0.95);

            std::vector<int> cQ = unscale_midpoint(InputOutput::index_to_coords(i, dimensions, scaledSize));
            data.emplace_back(prob, cQ);

            // optional wraparound for Y-dimension
            cQ[1] -= dimensionSize;
            data.emplace_back(prob, cQ);
            cQ[1] += 2 * dimensionSize;
            data.emplace_back(prob, cQ);
        }

        int neigh = std::clamp(0.01 * data.size(), 2.0, 8.0);
        maper = new IDW(data, neigh, 2, 0.1);
    }
}

// Modified get_value_at to fallback to prior if IDW is not ready
double StochasticQueryModel::get_value_at(const std::vector<int>& query) {
    if (!maper) return priorProb; // fallback
    double val = maper->predict(query);
    return val;
}