
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

double FunctionSpace::computeMeanHelper(std::vector<int>& query, int index, double& sum, int& count) const {
    if (index == query.size()) {
        sum += this->get(query);
        count++;
        return sum;
    }
    for (int i = 0; i < dimensionSize; i++) {
        query[index] = i;
        computeMeanHelper(query, index + 1, sum, count);
    }
    return sum;
}

double FunctionSpace::getRealMean() const {
    std::vector<int> query(dimensions, 0);
    double sum = 0.0;
    int count = 0;
    computeMeanHelper(query, 0, sum, count);
    return count > 0 ? sum / count : 0.0;
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

void FunctionSpace::resetResults(){
    results = Results();
}

void FunctionSpace::updateResults(double predicted, double actual){
    results.updateResults(actual, predicted);
}

void FunctionSpace::submitResults(){
    allResults.emplace_back(results);
}