#include "FunctionSpace.hpp"

#include <exception>
#include <queue>

FunctionSpace::FunctionSpace(int dimensions, int dimensionSize, SpaceFunctionType spaceFunction) : 
    StateSpace(dimensions, dimensionSize), 
    spaceFunction(spaceFunction) {}

int FunctionSpace::get_dimensions() const {
    return this->dimensions;
}

int FunctionSpace::get_dimension_size() const {
    return this->dimensionSize;
}

double FunctionSpace::get(const std::vector<int>& coords) const {
    //for (auto i : coords)
        //std::cout << i << " ";
    //std::cout << std::endl;
    if (coords.size() != dimensions)
        throw std::out_of_range("query dimension size is out of range");
    
    for (auto val : coords)
        if (val < 0 || val >= dimensionSize)
            throw std::out_of_range("query values are out of range");
    
    return this->spaceFunction(coords);
}


void FunctionSpace::getResultsHelper(Model& model, std::vector<int>& query, int index) {
    std::cout << "\n";

    std::cout << index << ": [";
    for (int i = 0; i < query.size() - 1; i++) {
        std::cout << query[i] << ", ";
    }
    std::cout << query[query.size()-1] << "] \n";
    
    if (index == query.size()) {
        return;
    }

    for (int i = 1; i < dimensionSize; i++) {
        query[index] = i;

        double actualResult = this->get(query);
        double predictedResult = model.get_value_at(query);
        std::cout << "result: " << actualResult << " - predicted: " << predictedResult << "\n";

        this->results.updateResults(actualResult, predictedResult);

        getResultsHelper(model, query, index + 1);
    }
    
    return;
}

Results FunctionSpace::getResults(Model& model) {
    results = Results();

    std::vector<int> initialQuery(dimensions, 0);

    double actualResult = this->get(initialQuery);
    double predictedResult = model.get_value_at(initialQuery);

    this->results.updateResults(actualResult, predictedResult);

    getResultsHelper(model, initialQuery, 0);

    this->allResults.push_back(this->results);
    
    return this->results;
}