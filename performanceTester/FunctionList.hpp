#pragma once
#include<vector>
#include<math.h>

namespace testfunctions {

double ackleyFunction(const std::vector<int>& query){
    const double a = 20.0;
    const double b = 0.2;
    const double c = 2 * M_PI;

    size_t n = query.size();
    if (n == 0) return 0.0;

    double sumSquares = 0.0;
    double sumCos = 0.0;

    for (int x : query) {
        double xd = static_cast<double>(x);  // convert int to double
        sumSquares += xd * xd;
        sumCos += cos(c * xd);
    }

    double term1 = -a * exp(-b * sqrt(sumSquares / n));
    double term2 = -exp(sumCos / n);

    return term1 + term2 + a + exp(1);
}

double sumpow (const std::vector<int>& query) {
    double sumSquares = 0.0;
    for (int i = 1; i <= query.size(); i++) {
        sumSquares += pow(abs(query[i - 1]),i);
    }
    return sumSquares;
}

double griewank(const std::vector<int>& query) {
    double sumTerm1 = 0.0;
    double sumTerm2 = 1.0;
    for (int i = 1; i <= query.size(); i++) {
        int x = query[i - 1];
        sumTerm1 += pow(x,2) / 4000;
        sumTerm2 *= cos(x/sqrt(i));
    }
    return sumTerm1 - sumTerm2 + 1;
}

double rastrigin(const std::vector<int>& query) {
    double dimTerm = query.size() * 10;
    double sumTerm = 0.0;
    for (int x : query) {
        sumTerm += pow(x,2) - 10 * cos(2* M_PI * x );
    }
    return dimTerm + sumTerm;
}
}