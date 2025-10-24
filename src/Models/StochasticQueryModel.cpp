#include "StochasticQueryModel.hpp"



StochasticQueryModel::StochasticQueryModel(int dimensions, int dimensionSize, int totalQueries) 
    : Model(dimensions, dimensionSize, totalQueries) {
    
    shouldQuery = 10;
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
        int x = rawQuery[i];

        // Using double arithmetic for the ratio, but logic mirrors integer boundary intervals
        // scaled index s satisfies: ceil(s / scaleRatio) <= x <= ceil((s+1)/scaleRatio) - 1

        // Estimate s from continuous space:
        int s = (int) std::floor(x * scaleRatio);

        // Correct for rare rounding edge cases near interval boundaries
        while (s > 0 && x < std::ceil(s / scaleRatio)) s--;
        while (s + 1 < scaledSize && x >= std::ceil((s + 1) / scaleRatio)) s++;

        // Clamp to valid range
        s = std::max(0, std::min(s, scaledSize - 1));

        scaled[i] = s;
    }
    return scaled;
}


std::vector<std::pair<int,int>> StochasticQueryModel::get_unscaled_int_bounds(const std::vector<int>& scaledQuery) const {
    std::vector<std::pair<int,int>> bounds;
    bounds.reserve(scaledQuery.size());

    for (size_t i = 0; i < scaledQuery.size(); ++i) {
        int s = scaledQuery[i];
        // compute integer bounds using ceil arithmetic
        int lo = (int) std::ceil( (double)s / scaleRatio );
        int hi = (int) std::ceil( (double)(s + 1) / scaleRatio ) - 1;

        // clamp to valid domain [0, dimensionSize-1]
        lo = std::max(0, std::min(lo, dimensionSize - 1));
        hi = std::max(0, std::min(hi, dimensionSize - 1));

        bounds.emplace_back(lo, hi);
    }
    return bounds;
}

std::vector<int> StochasticQueryModel::sample_unscaled_point_from_scaled(const std::vector<int>& scaledQuery) {
    auto bounds = get_unscaled_int_bounds(scaledQuery);
    std::vector<int> result(scaledQuery.size());

    // single RNG instance (thread-safety depends on your app)
    static std::mt19937_64 rng((unsigned) std::random_device{}());

    for (size_t i = 0; i < bounds.size(); ++i) {
        int lo = bounds[i].first;
        int hi = bounds[i].second;

        if (lo <= hi) {
            std::uniform_int_distribution<int> dist(lo, hi);
            result[i] = dist(rng);
        } else {
            // fallback: no integer in interval (rare). choose nearest integer to midpoint
            double lower = (double) scaledQuery[i] / scaleRatio;
            double upper = (double)(scaledQuery[i] + 1) / scaleRatio;
            int mid = (int) std::floor((lower + upper) / 2.0);
            mid = std::max(0, std::min(mid, dimensionSize - 1));
            result[i] = mid;
        }

        // sanity check (optional in debug builds)
        // assert( std::floor(result[i] * scaleRatio) == scaledQuery[i] );
    }
    return result;
}

std::vector<int> StochasticQueryModel::unscale_midpoint(const std::vector<int>& scaledQuery) const {
    std::vector<int> unscaled(scaledQuery.size());
    for (size_t i = 0; i < scaledQuery.size(); ++i) {
        int s = scaledQuery[i];

        // Compute integer bounds for this scaled cell
        int lo = (int) std::ceil((double)s / scaleRatio);
        int hi = (int) std::ceil((double)(s + 1) / scaleRatio) - 1;

        // Clamp to valid range
        lo = std::max(0, std::min(lo, dimensionSize - 1));
        hi = std::max(0, std::min(hi, dimensionSize - 1));

        // Midpoint of integer subspace
        unscaled[i] = (lo + hi) / 2;
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

    return sample_unscaled_point_from_scaled(currentQueryPoint);
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
            data.emplace_back(prob, unscale_midpoint(p.first));
        }
        if (maper) delete maper;
        double neigh = 0.1 * totalPoints;
        maper = new IDW(data, neigh, 2, 0.1);
    }
}

double StochasticQueryModel::get_value_at(const std::vector<int>& query) {
    if (maper) return maper->predict(query);
    return 0.0;
}