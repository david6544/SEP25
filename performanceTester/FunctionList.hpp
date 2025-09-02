#pragma once
#include<vector>
#include<math.h>

/*
* Namespace for holding state space functions
*/
namespace testfunctions {

// https://www.desmos.com/3d/tqc43wbixk
double ackleyFunction(const std::vector<int>& query){
    const double a = 200.0;
    const double b = 1.5;
    const double c = 0.2;
    const double m = 0.001;
    const double horizontalTranslation = 150;

    size_t n = query.size();
    
    if (n == 0) return 0.0;

    double sumSquares = 0.0;
    double sumCos = 0.0;

    for (int x : query) {
        double xd = static_cast<double>(x - horizontalTranslation);  // convert int to double
        sumSquares += xd * xd;
        sumCos += 2 * cos(c * x);
    }

    double term1 = -a * exp(-b * sqrt(sumSquares / n));
    double term2 = -exp(2 * sumCos / n);

    return term1 + term2 + a + exp(1);
}

// https://www.desmos.com/3d/zjdogfxabu
double sumpow (const std::vector<int>& query) {
    double sumSquares = 0.0;
    int horizontalTranslation = 150;
    for (int i = 1; i <= query.size(); i++) {
        int x = query[i-1];
        double xd = static_cast<double>(x - horizontalTranslation);  // convert int to double
        sumSquares += pow(abs(xd),query.size() + 1);
    }
    return sumSquares * 0.00004;
}

// https://www.desmos.com/3d/0u33nq3ubb
double griewank(const std::vector<int>& query) {
    double sumTerm1 = 0.0;
    double prodTerm = 1.0;
    double modifier = 0.06;
    int horizontalTranslation = 300;
    
    for (int i = 1; i <= query.size(); i++) {
        int x = query[i - 1];
        double xd = static_cast<double>(x);
        double xd2 = static_cast<double>((modifier * x) - horizontalTranslation);  // convert int to double
        sumTerm1 += pow(xd,2) / 4000;
        prodTerm *= cos((xd2)/sqrt(i));
    }
    return sumTerm1 - (40 * (prodTerm) + 1);
}

// https://www.desmos.com/3d/pkq78jraem
double rastrigin(const std::vector<int>& query) {
    double dimTerm = query.size() * 10;
    double sumTerm = 0.0;
    double c = 0.007;
    double modifier = 0.00004;

    for (int x : query) {
        double xd = static_cast<double>(x);
        sumTerm += pow((xd*modifier),2) - 12 * cos(2* M_PI * xd * c);
    }
    return dimTerm + sumTerm;
}

// https://www.desmos.com/3d/nyz2xlcdwn
double michalewicz(const std::vector<int>& query) {
    double a = 1.5;
    double c = 0.003;
    double d = 0.000065;
    double m = 5.5;

    double sumTerm = 0.0;

    for (int i = 1 ; i <= query.size(); i++) {
        double x = query[i-1];
        double xd = static_cast<double>(x);
        sumTerm += a * sin(c * xd) * pow(sin((i * d * pow(xd,2))/M_PI),2*m);
    }
    return -sumTerm;
}
}