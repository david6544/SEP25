#pragma once
#include<vector>
#include<math.h>

/*
* Namespace for holding state space functions
*/
namespace testfunctions {


int dimSize = 0;
   
double translateIntoHypercube(int x, double lowerBound, double upperBound) {
    double scale = (upperBound - lowerBound) / (dimSize - 1);
    return x * scale;
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

    return term1 + term2 + a + exp(1);
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
    return sumSquares;
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
    return sumTerm1 - prodTerm + 1;
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
    return dimTerm + sumTerm;
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

    return std::abs(sumTerm);
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
    return finalTerm;
}


// Zakharov Function
// Category: Plate Function
// Evaluated on hypercube x_i ∈ [-5,10] 
// Dimensions: n
// https://www.sfu.ca/~ssurjano/zakharov.html
double zakharov(const std::vector<int>& query) {

    double d = query.size();
    
    double sumTerm1 = 0.0;
    double sumTerm2 = 0.0;
    double sumTerm3 = 0.0;

    for (int i = 1; i <= query.size(); i++) {
        double xd = testfunctions::translateIntoHypercube(query[i-1], -5, 10);
        sumTerm1 += pow(xd,2);
        sumTerm2 += 0.5 * i * xd;
        sumTerm3 += 0.5 * i * xd;
    }
    return sumTerm1 + pow(sumTerm2,2) + pow(sumTerm3, 4);
}


// Dixon-Price Function
// Category: Valley Function
// Evaluated on hypercube x_i ∈ [-10,10] 
// Dimensions: n
// https://www.sfu.ca/~ssurjano/dixonpr.html
double dixonPrice(const std::vector<int>& query) {

    double d = query.size();
    
    double sumTerm = 0.0;
    double initalTerm = pow(query[0] - 1, 2);


    for (int i = 2; i <= query.size(); i++) {
        double xd = testfunctions::translateIntoHypercube(query[i-1], -10, 10);
        double xdPrior = testfunctions::translateIntoHypercube(query[i-2], -10, 10);
        sumTerm += i * pow((2 * pow(xd,2) - xdPrior),2);
    }
    return initalTerm + sumTerm;
}



// Rosenbrock Function
// Category: Valley Function
// Evaluated on hypercube x_i ∈ [-5,10] 
// Dimensions: n
// https://www.sfu.ca/~ssurjano/rosen.html
double rosenbrock(const std::vector<int>& query) {
    
    double sumTerm1 = 0.0;

    for (int i = 1; i < query.size(); i++) {
        double xd = testfunctions::translateIntoHypercube(query[i-1], -5, 10);
        double xdPost = testfunctions::translateIntoHypercube(query[i], -5, 10);
        sumTerm1 += 100 * pow(xdPost - pow(xd,2), 2) + pow(xd - 1, 2);
    }
    return sumTerm1;
}


// Rotated Hyper Ellispoid Function
// Category: Bowl Function
// Evaluated on hypercube x_i ∈ [-65.536,65.536] 
// Dimensions: n
// https://www.sfu.ca/~ssurjano/rothyp.html
double hyperEllipsoid(const std::vector<int>& query) {
    
    double sumTerm = 0.0;

    for (int i = 0; i < query.size(); i++) {
        for (int j = 0; j < i; j++) {
            sumTerm += pow(query[j],2);
        }
    }

    return sumTerm;
}
}