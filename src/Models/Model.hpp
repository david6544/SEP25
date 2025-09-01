#ifndef MODEL_H
#define MODEL_H
#include <vector>

class Model {
protected:
    int dimensions, dimensionSize, totalQueries, currentQuery;
public:
        Model(int dimensions, int dimensionSize, int totalQueries): 
            totalQueries(totalQueries), dimensions(dimensions), dimensionSize(dimensionSize) {};
        int get_dimensions(){ return dimensions; }
        int get_dimensionSize(){ return dimensionSize; }
        virtual std::vector<int> get_next_query() = 0;
        virtual void update_prediction(const std::vector<int> &query, double result) = 0;
        virtual double get_value_at(const std::vector<int> &query) = 0;
};


#endif // MODEL_H