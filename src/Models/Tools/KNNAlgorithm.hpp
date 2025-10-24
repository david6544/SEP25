#pragma once
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <limits>

struct KDNode {
    std::vector<int> point;
    double value;
    KDNode* left;
    KDNode* right;

    KDNode(const std::vector<int>& pt, double val)
        : point(pt), value(val), left(nullptr), right(nullptr) {}
};

class KNNTree {
public:
    struct KDResult {
        double distance;
        double value;
        std::vector<int> point;
    };

    KNNTree(const std::vector<std::pair<double, std::vector<int>>>& data) {
        root = build(data, 0);
    }

    ~KNNTree() {
        destroy(root);
    }

    std::vector<KDResult> getKNearest(const std::vector<int>& query, int K) const {
        if (!root || K <= 0) return {}; // no tree or invalid K

        std::priority_queue<std::pair<double, KDNode*>> best; // (distance, node)
        search(root, query, K, 0, best);

        std::vector<KDResult> result;
        result.reserve(std::min(K, (int)best.size()));

        while (!best.empty()) {
            auto [dist, node] = best.top();
            best.pop();
            result.push_back({ dist, node->value, node->point });
        }

        std::reverse(result.begin(), result.end());
        return result;
    }




private:
    KDNode* root = nullptr;

    static double distance(const std::vector<int>& a, const std::vector<int>& b) {
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
            sum += (a[i] - b[i]) * (a[i] - b[i]);
        return std::sqrt(sum);
    }

    KDNode* build(std::vector<std::pair<double, std::vector<int>>> data, int depth) {
        if (data.empty()) return nullptr;

        size_t axis = depth % data[0].second.size();
        std::sort(data.begin(), data.end(),
            [axis](const auto& a, const auto& b) {
                return a.second[axis] < b.second[axis];
            });

        size_t median = data.size() / 2;
        KDNode* node = new KDNode(data[median].second, data[median].first);

        std::vector<std::pair<double, std::vector<int>>> left(data.begin(), data.begin() + median);
        std::vector<std::pair<double, std::vector<int>>> right(data.begin() + median + 1, data.end());

        node->left = build(left, depth + 1);
        node->right = build(right, depth + 1);
        return node;
    }

    void search(KDNode* node, const std::vector<int>& query, int K, int depth,
                std::priority_queue<std::pair<double, KDNode*>>& best) const {
        if (!node) return;

        double dist = distance(query, node->point);
        if ((int)best.size() < K)
            best.push({ dist, node });
        else if (dist < best.top().first) {
            best.pop();
            best.push({ dist, node });
        }

        size_t axis = depth % query.size();
        KDNode* next = (query[axis] < node->point[axis]) ? node->left : node->right;
        KDNode* other = (query[axis] < node->point[axis]) ? node->right : node->left;

        search(next, query, K, depth + 1, best);

        // Check if we need to search the other side
        if ((int)best.size() < K || std::abs(query[axis] - node->point[axis]) < best.top().first)
            search(other, query, K, depth + 1, best);
    }

    void destroy(KDNode* node) {
        if (!node) return;
        destroy(node->left);
        destroy(node->right);
        delete node;
    }
};
