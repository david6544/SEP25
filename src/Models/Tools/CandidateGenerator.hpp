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

static thread_local std::mt19937_64 rng(std::random_device{}());

class CandidateGenerator {
private:
    int dims;
    CandidateNode* root;
    const std::vector<std::vector<int>> dimLimits;

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

        if (idx < query.size() - 1) {
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
            if (itemPos != node->itemNode.size() - 1) {
                node->itemNode[itemPos] = lastPair;
                node->itemIdx[lastPair.first] = itemPos;
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
        std::vector<std::vector<int>> candidates(count, std::vector<int>(dims));
        std::vector<CandidateNode*> runners(count, root);

        for (int i = 0; i < dims; i++) {
            int dimMin = dimLimits[i][0];
            int dimRange = dimLimits[i][1] - dimLimits[i][0] + 1;

            for (int j = 0; j < count; j++) {
                CandidateNode* node = runners[j];

                // fallback random value if node empty
                if (!node || node->itemNode.empty()) {
                    candidates[j][i] = dimMin + (rng() % dimRange);
                    continue;
                }

                // random index in available nodes
                int randIdx = rng() % node->itemNode.size();
                auto &p = node->itemNode[randIdx];

                candidates[j][i] = dimMin + p.first;
                runners[j] = p.second;
            }
        }

        return candidates;
    }

};

