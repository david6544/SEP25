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
    for (auto i : coords)
        std::cout << i << " ";
    std::cout << std::endl;
    if (coords.size() != dimensions)
        throw std::out_of_range("query dimension size is out of range");
    
    for (auto val : coords)
        if (val < 0 || val >= dimensionSize)
            throw std::out_of_range("query values are out of range");
    
    return this->spaceFunction(coords);
}



void FunctionSpace::getResultsHelper(Model& model, std::vector<int>& query, int dimension){
    for (auto i : query)
        std::cout << i << " ";
    std::cout << std::endl;
    
    if (dimension == query.size()){
        return;
    }

    for (int i = 1; i < dimensionSize; i++){
        query[dimension] = i;

        double actualResult = this->get(query);
        double predictedResult = model.get_value_at(query);

        this->results.updateResults(actualResult, predictedResult);

        getResultsHelper(model, query, dimension+1);
    }
    query[dimension] = 0;
    
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