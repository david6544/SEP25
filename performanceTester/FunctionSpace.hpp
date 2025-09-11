#ifndef FUNCTION_SPACE
#define FUNCTION_SPACE

#include "../src/StateSpace/ArrayStateSpace.hpp"
#include "../src/Models/Model.hpp"
#include <vector>
#include <functional>
#include <algorithm>
#include <limits>

#define SpaceFunctionType std::function<double(const std::vector<int>&)>

typedef struct Results {
    int correct = 0;
    int totalSeen = 0;

    double sumAbsoluteError = 0;
    double sumSquaredError = 0;

    double predictedResultSum = 0;

    double minActual = std::numeric_limits<double>::max();
    double maxActual = std::numeric_limits<double>::lowest();
    double minPredicted = std::numeric_limits<double>::max();
    double maxPredicted = std::numeric_limits<double>::lowest();
    double realMean = 0.0;
    void updateResults(double actual, double predicted){
        totalSeen++;
        if (predicted >= actual * 0.99 && predicted <= actual * 1.01){
            this->correct++;
        }
        double error = abs(actual-predicted);
        sumAbsoluteError += error;
        sumSquaredError += pow(error, 2);

        minActual = std::min(minActual, actual);
        maxActual = std::max(maxActual, actual);

        minPredicted = std::min(minPredicted, predicted);
        maxPredicted = std::max(maxPredicted, predicted);

        predictedResultSum += predicted;
    }
    double percentCorrect() const {
        return totalSeen > 0 ? static_cast<double>(correct) / totalSeen * 100.0 : 0.0;
    }

    double meanAbsoluteError() const {
        return totalSeen > 0 ? sumAbsoluteError / totalSeen : 0.0;
    }

    double rootMeanSquaredError() const {
        return totalSeen > 0 ? std::sqrt(sumSquaredError / totalSeen) : 0.0;
    }

    double meanPredicted() const {
        return totalSeen > 0 ? predictedResultSum / totalSeen : 0.0;
    }
} Results;

class FunctionSpace : public StateSpace {
private:
    SpaceFunctionType spaceFunction;
    Results results;

    std::vector<Results> allResults;
    
    void getResultsHelper(Model& model, std::vector<int>& query, int dimension);
    double computeMeanHelper(std::vector<int>& query, int index, double& sum, int& count) const;
public:
    FunctionSpace(int dimensions, int dimensionSize, SpaceFunctionType spaceFunction);

    int get_dimensions() const override;

    int get_dimension_size() const override;
    
    double get(const std::vector<int>& coords) const override;

    Results getResults(Model& model);
    std::vector<Results> getAllResults(){ return allResults; };

    double getRealMean() const;
};


#endif //FUNCTION_SPACE