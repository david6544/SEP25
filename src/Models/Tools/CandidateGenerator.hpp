#pragma once

#include <vector>
#include <unordered_map>
#include <random>
#include <algorithm>
#include <stdexcept>

#include <iostream>

struct CandidateNode {
    std::unordered_map<int, int> itemIdx;
    std::vector<std::pair<int, CandidateNode*>> itemNode;
};

class CandidateGenerator {
private:
    int dims;
    CandidateNode* root;
    const std::vector<std::vector<int>> dimLimits;
    std::mt19937 rng{std::random_device{}()};

    CandidateNode* initDefaultNode(int idx) {
        CandidateNode* node = new CandidateNode();
        int r = dimLimits[idx][1] - dimLimits[idx][0] + 1;
        node->itemNode.reserve(r);
        for (int i = 0; i < r; i++) {
            int actualValue = dimLimits[idx][0] + i;
            node->itemNode.push_back({i, nullptr});
            node->itemIdx[actualValue] = i;
        }
        return node;
    }

    void freeNode(CandidateNode* node) {
        if (!node) return;
        for (auto &kv : node->itemNode) {
            freeNode(kv.second);
        }
        delete node;
    }

    bool addHelper(const std::vector<int>& query, CandidateNode* node, int idx) {
        auto it = node->itemIdx.find(query[idx]);
        if (it == node->itemIdx.end()) {
            return false;
        }
        int itemPos = it->second;
        auto &itemPair = node->itemNode[itemPos];

        bool removeCurrent = false;

        if (idx < (int)query.size() - 1) {
            if (itemPair.second) {
                bool childEmpty = addHelper(query, itemPair.second, idx + 1);
                if (childEmpty) {
                    freeNode(itemPair.second);
                    itemPair.second = nullptr;
                    removeCurrent = true;
                }
            }
        } else {
            removeCurrent = true;
        }

        if (removeCurrent) {
            auto lastPair = node->itemNode.back();
            int lastValue = dimLimits[idx][0] + lastPair.first;
            if (itemPos != (int)node->itemNode.size() - 1) {
                node->itemNode[itemPos] = lastPair;
                node->itemIdx[lastValue] = itemPos;
            }
            node->itemNode.pop_back();
            node->itemIdx.erase(query[idx]);
        }

        return node->itemNode.empty();
    }



public:
    CandidateGenerator(int dims, std::vector<std::vector<int>> dimLimits)
        : dims(dims), dimLimits(dimLimits)
    {
        root = initDefaultNode(0);
    }

    ~CandidateGenerator() { freeNode(root); }
    

    void addQueriedPoint(const std::vector<int>& query){
        addHelper(query, root, 0);
    }

    std::vector<std::vector<int>> getCandidates(int count) {
        std::vector<std::vector<int>> candidates(count);
        std::vector<CandidateNode*> runners(count, root);

        for (int i = 0; i < dims; i++){
            std::uniform_int_distribution<int> limitDist(dimLimits[i][0], dimLimits[i][1]);
            for (int j = 0; j < count; j++){
                if (runners[j] == nullptr || runners[j]->itemIdx.size() == 0){
                    candidates[j].push_back(limitDist(rng));
                    continue;
                }
                std::uniform_int_distribution<int> dist(0, (int)runners[j]->itemNode.size() - 1);
                int randIdx = dist(rng);
                auto p = runners[j]->itemNode[randIdx];
                runners[j] = p.second;
                candidates[j].push_back(dimLimits[i][0] + p.first);
            }
        }

        return candidates;
    }
};

