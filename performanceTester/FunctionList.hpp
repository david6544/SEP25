#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>
#include <limits>
#include <climits>

#include "../src/InputOutput/InputOutput.hpp"

#define SpaceFunctionType std::function<double(const std::vector<int>&)>


/*
* Namespace for holding state space functions
*/
namespace testfunctions {

int dimSize = 0;
int dims = 0;
std::unordered_map<std::string, std::pair<double,double>> minMaxCache;
bool lock = false;

std::pair<double, double> getMinMax(SpaceFunctionType function, std::string functionId){
    if (minMaxCache.find(functionId) != minMaxCache.end()){
        return minMaxCache[functionId];
    }

    if (lock){
        return {-1, -1};
    }
    lock = true;

    int maxIdx = pow(dimSize, dims);
    std::pair<double, double> minMax = {std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest()};
    for (int i = 0; i < maxIdx; i++){
        std::vector<int> coords = InputOutput::index_to_coords(i, dims, dimSize);
        double output = function(coords);
        minMax.first = std::min(minMax.first, output);
        minMax.second = std::max(minMax.second, output);
    }

    minMaxCache.insert({functionId, minMax});
    lock = false;
    return minMax;
}

// Translate discrete query into continuous hypercube
double translateIntoHypercube(int x, double lowerBound, double upperBound) {
    if (dimSize <= 1) return lowerBound;
    double scale = (upperBound - lowerBound) / (dimSize - 1);
    return lowerBound + x * scale;
}

// Generic normalization
double normalize(double value, std::string functionId) {
    if (minMaxCache.find(functionId) != minMaxCache.end()){
        auto minMax = minMaxCache[functionId];
        if (minMax.second - minMax.first < 1) 
            return std::clamp((value - minMax.first), 0.0, 1.0);
        return std::clamp((value - minMax.first) / (minMax.second - minMax.first), 0.0, 1.0);
    }

    return value;
}

// ------------------ Test Functions ------------------

// Ackley Function
double ackleyFunction(const std::vector<int>& query) {
    getMinMax(ackleyFunction, "ackleyFunction");
    const double a = 20.0, b = 0.2, c = 2 * M_PI;
    size_t d = query.size();
    if (d == 0) return 0.0;

    double sumSquares = 0.0, sumCos = 0.0;
    for (auto& x : query) {
        double xd = translateIntoHypercube(x, -32.768, 32.768);
        sumSquares += xd * xd;
        sumCos += cos(c * xd);
    }
    double raw = -a * exp(-b * sqrt(sumSquares / d)) - exp(sumCos / d) + a + M_E;
    return normalize(raw, "ackleyFunction"); // empirically safe
}

// Sum of Different Powers
double sumpow(const std::vector<int>& query) {
    getMinMax(sumpow, "sumpow");
    size_t d = query.size();
    if (d == 0) return 0.0;

    double sum = 0.0;
    for (size_t i = 0; i < d; ++i) {
        double xd = translateIntoHypercube(query[i], -1, 1);
        sum += pow(std::abs(xd), d + 1);
    }
    return normalize(sum, "sumpow");
}

// Griewank
double griewank(const std::vector<int>& query) {
    getMinMax(griewank,"griewank");
    size_t d = query.size();
    if (d == 0) return 0.0;
    double sumTerm = 0.0, prodTerm = 1.0;
    for (size_t i = 0; i < d; ++i) {
        double xd = translateIntoHypercube(query[i], -600, 600);
        sumTerm += xd * xd / 4000.0;
        prodTerm *= cos(xd / sqrt(i + 1));
    }
    double raw = sumTerm - prodTerm + 1;
    return normalize(raw, "griewank");
}

// Rastrigin
double rastrigin(const std::vector<int>& query) {
    getMinMax(rastrigin,"rastrigin");
    size_t d = query.size();
    if (d == 0) return 0.0;
    double sum = 0.0;
    for (auto x : query) {
        double xd = translateIntoHypercube(x, -5.12, 5.12);
        sum += xd * xd - 10 * cos(2 * M_PI * xd);
    }
    double raw = 10.0 * d + sum;
    return normalize(raw, "rastrigin");
}

// Michalewicz
double michalewicz(const std::vector<int>& query) {
    getMinMax(michalewicz,"michalewicz");
    size_t d = query.size();
    if (d == 0) return 0.0;
    double sum = 0.0, m = 10.0;
    for (size_t i = 0; i < d; ++i) {
        double xd = translateIntoHypercube(query[i], 0, M_PI);
        sum += sin(xd) * pow(sin((i + 1) * xd * xd / M_PI), 2 * m);
    }
    return normalize(std::abs(sum), "michalewicz");
}

// Power Sum
double powerSum(const std::vector<int>& query) {
    getMinMax(powerSum,"powerSum");
    size_t d = query.size();
    if (d == 0) return 0.0;
    std::vector<int> b = {0,8,18,44,114,223,556,1243};
    double finalTerm = 0.0;

    for (size_t j = 0; j < d && j < b.size(); ++j) {
        double sumTerm = 0.0;
        for (size_t i = 0; i < d; ++i) {
            double xd = translateIntoHypercube(query[i], 0, d);
            sumTerm += pow(xd, j + 1);
        }
        sumTerm -= b[j];
        finalTerm += pow(sumTerm, 2);
    }
    return normalize(finalTerm, "powerSum");
}

// Zakharov
double zakharov(const std::vector<int>& query) {
    getMinMax(zakharov,"zakharov");
    size_t d = query.size();
    if (d == 0) return 0.0;
    double sum1 = 0.0, sum2 = 0.0;
    for (size_t i = 0; i < d; ++i) {
        double xd = translateIntoHypercube(query[i], -5, 10);
        sum1 += xd * xd;
        sum2 += 0.5 * (i + 1) * xd;
    }
    double raw = sum1 + pow(sum2, 2) + pow(sum2, 4);
    return normalize(raw, "zakharov");
}

// Dixon-Price
double dixonPrice(const std::vector<int>& query) {
    getMinMax(dixonPrice,"dixonPrice");
    size_t d = query.size();
    if (d == 0) return 0.0;
    double sum = pow(translateIntoHypercube(query[0], -10, 10) - 1, 2);
    for (size_t i = 1; i < d; ++i) {
        double xd = translateIntoHypercube(query[i], -10, 10);
        double xdPrior = translateIntoHypercube(query[i - 1], -10, 10);
        sum += (i + 1) * pow(2 * xd * xd - xdPrior, 2);
    }
    return normalize(sum, "dixonPrice");
}

// Rosenbrock
double rosenbrock(const std::vector<int>& query) {
    getMinMax(rosenbrock,"rosenbrock");
    size_t d = query.size();
    if (d < 2) return 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < d - 1; ++i) {
        double xd = translateIntoHypercube(query[i], -5, 10);
        double xdNext = translateIntoHypercube(query[i + 1], -5, 10);
        sum += 100 * pow(xdNext - xd * xd, 2) + pow(xd - 1, 2);
    }
    return normalize(sum, "rosenbrock");
}

// Rotated Hyper Ellipsoid
double hyperEllipsoid(const std::vector<int>& query) {
    getMinMax(hyperEllipsoid,"hyperEllipsoid");
    size_t d = query.size();
    if (d == 0) return 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < d; ++i)
        for (size_t j = 0; j <= i; ++j) {
            double xd = translateIntoHypercube(query[j], -65.536, 65.536);
            sum += xd * xd;
        }
    return normalize(sum, "hyperEllipsoid");
}

} // namespace testfunctions
