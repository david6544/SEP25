#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <functional>

struct CandidateNode {
    std::unordered_map<int, CandidateNode*> usedMap;
    std::unordered_map<int, int> unusedMap;
    std::vector<int> unused, used;
};

class CandidateGenerator {
private:
    int dims;
    CandidateNode* root;
    std::mt19937 rng{std::random_device{}()};

    CandidateNode* initDefaultNode(int dimSize) {
        CandidateNode* newNode = new CandidateNode();
        newNode->unused.reserve(dimSize);
        for (int i = 0; i < dimSize; i++) {
            newNode->unused.push_back(i);
            newNode->unusedMap[i] = i;
        }
        return newNode;
    }

    void freeNode(CandidateNode* node) {
        for (auto& kv : node->usedMap) {
            freeNode(kv.second);
        }
        delete node;
    }

public:
    CandidateGenerator(int dims, int dimSize) : dims(dims) {
        root = initDefaultNode(dimSize);
    }

    ~CandidateGenerator() { freeNode(root); }

    void addCandidate(const std::vector<int>& query) {
        if ((int)query.size() != dims) {
            throw std::out_of_range("[addCandidate] query size not the same");
        }

        CandidateNode* runner = root;
        for (int i = 0; i < dims; i++) {
            int val = query[i];
            if (runner->usedMap.find(val) == runner->usedMap.end()) {
                auto it = runner->unusedMap.find(val);
                if (it == runner->unusedMap.end()) {
                    throw std::runtime_error("[addCandidate] value not in unusedMap");
                }

                // Move value from unused -> used
                int idx = it->second;
                runner->unused[idx] = runner->unused.back();
                runner->unusedMap[runner->unused[idx]] = idx;
                runner->unused.pop_back();
                runner->unusedMap.erase(val);

                runner->used.push_back(val);
                runner->usedMap[val] = initDefaultNode(runner->unused.size() + runner->used.size());
            }
            runner = runner->usedMap[val];
        }
    }

    // Generate unique candidates within dimensionLimits
    std::vector<std::vector<int>> getCandidates(
        const std::vector<std::vector<int>>& dimensionLimits,
        int maxCount)
    {
        std::vector<std::vector<int>> results;
        std::vector<int> current;

        std::function<void(CandidateNode*, int)> helper = [&](CandidateNode* node, int depth) {
            if (depth == dims) {
                results.push_back(current);
                return;
            }
            if ((int)results.size() >= maxCount) return;

            // Filter unused values by dimensionLimits
            std::vector<int> options;
            for (int val : node->unused) {
                if (val >= dimensionLimits[depth][0] && val <= dimensionLimits[depth][1]) {
                    options.push_back(val);
                }
            }

            std::shuffle(options.begin(), options.end(), rng);

            for (int val : options) {
                if ((int)results.size() >= maxCount) break;

                current.push_back(val);

                CandidateNode* nextNode = nullptr;
                auto it = node->usedMap.find(val);
                if (it != node->usedMap.end()) {
                    nextNode = it->second;
                } else {
                    // Temporary node for non-destructive exploration
                    nextNode = initDefaultNode(node->unused.size() + node->used.size());
                }

                helper(nextNode, depth + 1);

                if (it == node->usedMap.end()) delete nextNode;
                current.pop_back();
            }
        };

        helper(root, 0);
        return results;
    }
};
