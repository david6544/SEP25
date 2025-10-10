#include <vector>

class Mapping {
protected:
    std::vector<std::pair<double, std::vector<int>>> queriedPoints;
public:
    Mapping(std::vector<std::pair<double, std::vector<int>>> queriedPoints) : queriedPoints(queriedPoints) {}

    virtual double predict(std::vector<int> query);
};