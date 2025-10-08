#include "KDTreeStateSpace.hpp"

// Constructor
KDTreeStateSpace::KDTreeStateSpace(int dimensions, int dimensionSize)
    : StateSpace(dimensions, dimensionSize), root(nullptr) {
        // Weigh max depth by number of dimensions and dimension size.


        // Can also just choose a fixed max depth. This is a heuristic.
        // Setting maxDepth to -1 means no limit.
        int totalPoints = 1;
        for (int d = 0; d < dimensions; ++d)
            totalPoints *= dimensionSize;

        this->maxDepth = static_cast<int>(std::ceil(std::log2(totalPoints))) * 1.2; // 20% deeper

        std::cout << "KDTree max depth set to " << this->maxDepth << "\n";
    }

// Destructor
KDTreeStateSpace::~KDTreeStateSpace() {
    destroy(root);
}

// Recursively free nodes
void KDTreeStateSpace::destroy(TreeNode* node) {
    if (!node) return;

    // Recursively free all bucketed children
    for (auto* child : node->bucket) {
        destroy(child);
    }

    destroy(node->left);
    destroy(node->right);
    delete node;
}


// Dimensions accessors
int KDTreeStateSpace::get_dimensions() const {
    return dimensions;
}

int KDTreeStateSpace::get_dimension_size() const {
    return dimensionSize;
}

// Public insert entry point
double KDTreeStateSpace::insert(const std::vector<int>& coords, double value) {
    if (coords.size() != static_cast<size_t>(dimensions)) {
        throw std::invalid_argument("Coordinate dimensionality mismatch");
    }
    root = insert_recursive(root, coords, value, 0);
    return value;
}

// Recursive insert
KDTreeStateSpace::TreeNode* KDTreeStateSpace::insert_recursive(
    TreeNode* node, const std::vector<int>& coords, double value, int depth) 
{
    int axis = depth % this->dimensions;
    // It may also be beneficial here to do depth limit and bucketing.
    if (!node) {
        return new TreeNode(coords, value, axis);
    }


    if (maxDepth >= 0 && depth >= maxDepth) {
        // Check if point already exists in bucket
        std::cout << "Max depth reached at depth " << depthReached << "\n";
        depthReached++;
        for (auto* child : node->bucket) {
            if (child->coords == coords) {
                child->value = value;
                return node;
            }
        }
        // Add to bucket
        if (!node->isBucket) {
            node->isBucket = true;
        }
        TreeNode* newNode = new TreeNode(coords, value, depth % dimensions);
        node->bucket.push_back(newNode);
        return node;
    }


    if (coords == node->coords) {
        // If point already exists, update its value
        node->value = value;
    } else if (coords[axis] < node->coords[axis]) {
        node->left = insert_recursive(node->left, coords, value, depth + 1);
    } else {
        node->right = insert_recursive(node->right, coords, value, depth + 1);
    }

    return node;
}

// Public get entry point
double KDTreeStateSpace::get(const std::vector<int>& coords) const {
    if (coords.size() != static_cast<size_t>(dimensions)) {
        throw std::invalid_argument("Coordinate dimensionality mismatch");
    }

    try {
        return get_recursive(root, coords, 0);
    } catch (std::out_of_range&) {
        // Return default value if not present
        return 0.0;
    }
}


// Recursive search
double KDTreeStateSpace::get_recursive(TreeNode* node, const std::vector<int>& coords, int depth) const {
    if (!node) throw std::out_of_range("Coordinates not found in KDTree");

    // If this is a bucket node, check all children
    if (node->isBucket) {
        for (auto* child : node->bucket) {
            if (child->coords == coords) {
                return child->value;
            }
        }
        throw std::out_of_range("Coordinates not found in bucket");
    }

    if (coords == node->coords) return node->value;

    int axis = depth % dimensions;
    if (coords[axis] < node->coords[axis]) {
        return get_recursive(node->left, coords, depth + 1);
    } else {
        return get_recursive(node->right, coords, depth + 1);
    }
}


KDTreeStateSpace::TreeNode* KDTreeStateSpace::get_root() {
    return this->root;
}


// Compute squared Euclidean distance (faster than sqrt)
double KDTreeStateSpace::squared_distance(const std::vector<int>& a,
                                          const std::vector<int>& b) const {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double diff = static_cast<double>(a[i] - b[i]);
        sum += diff * diff;
    }
    return sum;
}

// Public entry point
std::vector<KDTreeStateSpace::TreeNode*> 
KDTreeStateSpace::nearest_neighbor(const std::vector<int>& coords) const {
    if (coords.size() != static_cast<size_t>(dimensions)) {
        throw std::invalid_argument("Coordinate dimensionality mismatch");
    }

    if (!root) {
        // KDTree is empty, return empty vector
        return {};
    }

    TreeNode* nearestNode = nullptr;
    double bestDist = std::numeric_limits<double>::max();

    nearest_recursive(root, coords, 0, nearestNode, bestDist);

    if (!nearestNode) {
        throw std::out_of_range("KDTree is empty");
    }

    // If nearest node is a bucket container, return all nodes in the bucket
    if (nearestNode->isBucket) {
        return nearestNode->bucket;
    }

    // Otherwise return a single-node vector
    return { nearestNode };
}


// Recursive nearest search
void KDTreeStateSpace::nearest_recursive(TreeNode* node,
                                         const std::vector<int>& target,
                                         int depth,
                                         TreeNode*& bestNode,
                                         double& bestDist) const {
    if (!node) return;

    // If this node is a bucket container, stop here — this bucket is the closest
    if (node->isBucket) {
        for (auto* child : node->bucket) {
            double dist = squared_distance(target, child->coords);
            if (dist < bestDist) {
                bestDist = dist;
                bestNode = child;
            }
        }
        return; // no need to go further
    }


    double dist = squared_distance(target, node->coords);
    if (dist < bestDist) {
        bestDist = dist;
        bestNode = node;
    }

    int axis = depth % dimensions;
    TreeNode* near = (target[axis] < node->coords[axis]) ? node->left : node->right;
    TreeNode* far  = (target[axis] < node->coords[axis]) ? node->right : node->left;

    // Explore near side first
    nearest_recursive(near, target, depth + 1, bestNode, bestDist);

    double diff = static_cast<double>(target[axis] - node->coords[axis]);
    if (diff * diff < bestDist) {
        nearest_recursive(far, target, depth + 1, bestNode, bestDist);
    }
}
