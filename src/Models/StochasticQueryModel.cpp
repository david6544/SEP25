#include "StochasticQueryModel.hpp"



StochasticQueryModel::StochasticQueryModel(int dimensions, int dimensionSize, int totalQueries) 
    : Model(dimensions, dimensionSize, totalQueries) {
    qt = new QueryTree(dimensions, dimensionSize, 15);
    totalPoints = pow(dimensionSize, dimensions);
    shouldQuery = totalQueries/(totalPoints/2);
}

StochasticQueryModel::~StochasticQueryModel(){
    if (this->maper) delete this->maper;
    if (this->qt) delete this->qt;
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

    return currentQueryPoint;
}

void StochasticQueryModel::update_prediction(const std::vector<int>& query, double result) {
    // sanity: if query doesn't match currentQueryPoint, optionally handle it
    // (here we expect they match)
    if (query != currentQueryPoint) {
        // Optional: log or handle unexpected mismatch
        std::cerr << "Warning: update_prediction called with non-current query\n";
    }

    // increment per-point counters
    if (result >= 0.9) wCount++;
    tCount++;

    // When we've collected enough samples for this point, flush them to the map
    if (tCount >= shouldQuery) {
        auto &currentResults = queryWinTotal[query]; // creates if missing
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
        maper = new IDW(data, 0.01 * totalPoints, 3);
    }
}

double StochasticQueryModel::get_value_at(const std::vector<int>& query) {
    if (maper) return maper->predict(query);
    return 0.0;
}