#pragma once
#include<vector>
#include<math.h>
#include <algorithm>

/*
* Namespace for holding state space functions
*/
namespace testfunctions {


int dimSize = 0;
   
double translateIntoHypercube(int x, double lowerBound, double upperBound) {
    double scale = (upperBound - lowerBound) / (dimSize - 1);
    return lowerBound + x * scale;
}

inline double normalize(double value, double minVal, double maxVal) {
    return std::clamp((value - minVal) / (maxVal - minVal), 0.0, 1.0);
}

// Ackley Function
// Hypercube in x_i ∈ [-32.768, 32.768],
// Category: Many Local Minima
// Dimensions: n
// https://www.sfu.ca/~ssurjano/ackley.html
double ackleyFunction(const std::vector<int>& query){

    const double a = 20.0;
    const double b = 0.2;
    const double c = 2 * M_PI;

    size_t d = query.size();
    
    if (d == 0) return 0.0;

    double sumSquares = 0.0;
    double sumCos = 0.0;

    for (auto& x : query) {
        double xd = testfunctions::translateIntoHypercube(x, -32.768, 32.768);
        sumSquares += xd * xd;
        sumCos += cos(c * xd);
    }

    double term1 = -a * exp(-b * sqrt(sumSquares / d));
    double term2 = -exp( sumCos / d);

    double raw = term1 + term2 + a + std::exp(1);

    // Ackley min = 0, typical max ≈ 22
    return normalize(raw, 0.0, 22.0);
}


// Sum of Different Powers Function
// Hypercube in x_i  ∈ [-1, 1],
// Category: Bowl Shaped
// Dimensions: n
// https://www.sfu.ca/~ssurjano/sumpow.html
double sumpow (const std::vector<int>& query) {

    //translate query into the hypercube

    double sumSquares = 0.0;
    for (int i = 1; i <= query.size(); i++) {
        double xd = testfunctions::translateIntoHypercube(query[i-1], -1, 1);
        sumSquares += pow(abs(xd),query.size() + 1);
    }

    return normalize(sumSquares, 0.0, (double)query.size());
}

// Griewank function
// Category: Many Local Minima
// Hypercube around x_i ∈ [-600, 600]
// Dimensions: n
// https://www.sfu.ca/~ssurjano/griewank.html
double griewank(const std::vector<int>& query) {
    
    double sumTerm1 = 0.0;
    double prodTerm = 1.0;
    
    for (int i = 1; i <= query.size(); i++) {
        double xd = testfunctions::translateIntoHypercube(query[i-1],-600,600);
        sumTerm1 += pow(xd,2) / 4000;
        prodTerm *= cos((xd)/sqrt(i));
    }

    double raw = sumTerm1 - prodTerm + 1;
    // Griewank ∈ [0, ~2]
    return normalize(raw, 0.0, 2.0);

}

// Rastrigin function
// Category: Many local Minima
// Evaluated on hypercube x_i ∈ [-5.12, 5.12], 
// Dimensions: n
// https://www.sfu.ca/~ssurjano/rastr.html
double rastrigin(const std::vector<int>& query) {
    double dimTerm = query.size() * 10;
    double sumTerm = 0.0;

    for (int x : query) {
        double xd = testfunctions::translateIntoHypercube(x, -5.12, 5.12);
        sumTerm += pow((xd),2) - 10 * cos(2* M_PI * xd);
    }

    double raw = 10 * query.size() + sumTerm;
    // Rastrigin ∈ [0, ~200 * n]
    return normalize(raw, 0.0, 200.0 * query.size());
}

// Michalewicz function
// Category: Steep Ridges/Drops
// Evaluated on hypercube x_i ∈ [0, pi], 
// Dimensions: n
// https://www.sfu.ca/~ssurjano/michal.html
double michalewicz(const std::vector<int>& query) {

    double sumTerm = 0.0;
    double m = 10.0;

    for (int i = 1 ; i <= query.size(); i++) {
        double xd = testfunctions::translateIntoHypercube(query[i-1], 0, M_PI);
        sumTerm += sin(xd) * pow(sin((i * pow(xd,2))/M_PI),2*m);
    }

    // Michalewicz ∈ [0, n]
    return normalize(std::abs(sumTerm), 0.0, (double)query.size());
}

// Power Sum Function
// Category: Plate Function
// Evaluated on hypercube x_i ∈ [0, n], 
// Dimensions: n
// https://www.sfu.ca/~ssurjano/powersum.html
double powerSum(const std::vector<int>& query) {
    double d = query.size();
    
    double finalTerm = 0.0;

    std::vector<int> b = {0,8,18,44,114,223,556,1243};

    for (int j = 1; j <= query.size(); j++) {
        double sumTerm = 0.0;
        for (int i = 1; i <= query.size(); i++) {
            double xd = testfunctions::translateIntoHypercube(query[i-1], 0, d);
            sumTerm += pow(xd, j);
        }
        sumTerm -= b[j-1];
        finalTerm += pow(sumTerm,2);
    }

    // Empirically varies widely; scale by dimension²
    return normalize(finalTerm, 0.0, d * d * 100.0);
}


// Zakharov Function
// Category: Plate Function
// Evaluated on hypercube x_i ∈ [-5,10] 
// Dimensions: n
// https://www.sfu.ca/~ssurjano/zakharov.html
double zakharov(const std::vector<int>& query) {
    double sum1 = 0.0, sum2 = 0.0;
    for (int i = 1; i <= (int)query.size(); i++) {
        double xd = translateIntoHypercube(query[i - 1], -5, 10);
        sum1 += xd * xd;
        sum2 += 0.5 * i * xd;
    }
    double raw = sum1 + std::pow(sum2, 2) + std::pow(sum2, 4);
    // Zakharov ∈ [0, ~1000 * n]
    return normalize(raw, 0.0, 1000.0 * query.size());
}


// Dixon-Price Function
// Category: Valley Function
// Evaluated on hypercube x_i ∈ [-10,10] 
// Dimensions: n
// https://www.sfu.ca/~ssurjano/dixonpr.html
double dixonPrice(const std::vector<int>& query) {
    double sumTerm = std::pow(translateIntoHypercube(query[0], -10, 10) - 1, 2);
    for (int i = 2; i <= (int)query.size(); i++) {
        double xd = translateIntoHypercube(query[i - 1], -10, 10);
        double xdPrior = translateIntoHypercube(query[i - 2], -10, 10);
        sumTerm += i * std::pow((2 * xd * xd - xdPrior), 2);
    }
    // Dixon-Price ∈ [0, ~500 * n]
    return normalize(sumTerm, 0.0, 500.0 * query.size());
}



// Rosenbrock Function
// Category: Valley Function
// Evaluated on hypercube x_i ∈ [-5,10] 
// Dimensions: n
// https://www.sfu.ca/~ssurjano/rosen.html
double rosenbrock(const std::vector<int>& query) {
    double sumTerm = 0.0;
    for (int i = 0; i < (int)query.size() - 1; i++) {
        double xd = translateIntoHypercube(query[i], -5, 10);
        double xdPost = translateIntoHypercube(query[i + 1], -5, 10);
        sumTerm += 100 * std::pow(xdPost - xd * xd, 2) + std::pow(xd - 1, 2);
    }
    // Rosenbrock ∈ [0, ~5000 * n]
    return normalize(sumTerm, 0.0, 5000.0 * query.size());
}


// Rotated Hyper Ellispoid Function
// Category: Bowl Function
// Evaluated on hypercube x_i ∈ [-65.536,65.536] 
// Dimensions: n
// https://www.sfu.ca/~ssurjano/rothyp.html
double hyperEllipsoid(const std::vector<int>& query) {
    double sumTerm = 0.0;
    for (int i = 0; i < (int)query.size(); i++) {
        for (int j = 0; j <= i; j++) {
            double xd = translateIntoHypercube(query[j], -65.536, 65.536);
            sumTerm += xd * xd;
        }
    }
    // HyperEllipsoid ∈ [0, ~1e5 * n]
    return normalize(sumTerm, 0.0, 100000.0 * query.size());
}
}