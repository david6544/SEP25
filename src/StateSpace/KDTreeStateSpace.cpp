#include "KDTreeStateSpace.hpp"

// Constructor
KDTreeStateSpace::KDTreeStateSpace(int dimensions, int dimensionSize)
    : StateSpace(dimensions, dimensionSize), root(nullptr) {}

// Destructor
KDTreeStateSpace::~KDTreeStateSpace() {
    destroy(root);
}

// Recursively free nodes
void KDTreeStateSpace::destroy(TreeNode* node) {
    if (!node) return;
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

    if (!node) {
        return new TreeNode(coords, value, axis);
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
    return get_recursive(root, coords, 0);
}

// Recursive search
double KDTreeStateSpace::get_recursive(
    TreeNode* node, const std::vector<int>& coords, int depth) const 
{
    if (!node) {
        throw std::out_of_range("Coordinates not found in KDTree");
    }

    if (coords == node->coords) {
        return node->value;
    }

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
std::pair<std::vector<int>, double> 
KDTreeStateSpace::nearest_neighbor(const std::vector<int>& coords) const {
    if (coords.size() != static_cast<size_t>(dimensions)) {
        throw std::invalid_argument("Coordinate dimensionality mismatch");
    }

    std::vector<int> best_coords;
    double best_value = 0.0;
    double best_dist = std::numeric_limits<double>::max();

    nearest_recursive(root, coords, 0, best_coords, best_value, best_dist);

    if (best_coords.empty()) {
        throw std::out_of_range("KDTree is empty");
    }

    return {best_coords, best_value};
}

// Recursive nearest search
void KDTreeStateSpace::nearest_recursive(TreeNode* node,
                                         const std::vector<int>& target,
                                         int depth,
                                         std::vector<int>& best_coords,
                                         double& best_value,
                                         double& best_dist) const {
    if (!node) return;

    // Compute distance to this node
    double dist = squared_distance(target, node->coords);
    if (dist < best_dist) {
        best_dist = dist;
        best_coords = node->coords;
        best_value = node->value;
    }

    int axis = depth % dimensions;

    // Choose branch
    TreeNode* near = (target[axis] < node->coords[axis]) ? node->left : node->right;
    TreeNode* far  = (target[axis] < node->coords[axis]) ? node->right : node->left;

    // Search nearer side first
    nearest_recursive(near, target, depth + 1, best_coords, best_value, best_dist);

    // Check if we need to explore the other side
    double diff = static_cast<double>(target[axis] - node->coords[axis]);
    if (diff * diff < best_dist) {
        nearest_recursive(far, target, depth + 1, best_coords, best_value, best_dist);
    }
}
