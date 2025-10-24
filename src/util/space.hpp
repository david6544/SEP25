
#pragma once

#include "io.hpp"

#include <ranges>
#include <cmath>
#include <stack>
#include <span>
#include <queue>
#include <vector>

namespace skx::space::func {
    // (0,1)^2 -> (0,1)
    double ackley(double x, double y) {
        double x1 = 10.0 * x - 5.0;
        double y1 = 10.0 * y - 5.0;
        double term1 = -20.0 * std::exp(-0.2 * std::sqrt(0.5 * (x1 * x1 + y1 * y1)));
        double term2 = -std::exp(0.5 * (std::cos(2.0 * M_PI * x1) + std::cos(2.0 * M_PI * y1)));
        return (term1 + term2 + 20.0 + M_E) / 21.0;  // normalized to ~[0,1]
    }
}

namespace skx::space::euc {

template <std::ranges::random_access_range RangeCoord>
requires std::convertible_to<std::ranges::range_value_t<RangeCoord>, double>
double normSquared(const RangeCoord& coord) {
    double euc = 0;
    for (size_t i = 0; i < std::ranges::size(coord); i++) {
        euc += std::pow(static_cast<double>(coord[i]), 2.0);
    }
    return euc;
}

template <std::ranges::random_access_range RangeCoordA, std::ranges::random_access_range RangeCoordB>
requires std::convertible_to<std::ranges::range_value_t<RangeCoordB>, std::ranges::range_value_t<RangeCoordA>>
    && std::convertible_to<std::ranges::range_value_t<RangeCoordA>, double>
double normSquared(const RangeCoordA& coord_a, const RangeCoordB& coord_b) {
    double euc = 0.0;
    for (size_t i = 0; i < std::max(coord_a.size(),coord_b.size()); i++) {
        std::ranges::range_value_t<RangeCoordA> xa{};
        std::ranges::range_value_t<RangeCoordB> xb{};
        if (i < coord_a.size()) {
            xa = coord_a[i];
        }
        if (i < coord_b.size()) {
            xb = coord_b[i];
        }
        euc += std::pow(static_cast<double>(xa - xb), 2.0);
    }
    return euc;
}

template <std::ranges::random_access_range RangeCoord>
requires std::convertible_to<std::ranges::range_value_t<RangeCoord>, double>
double norm(const RangeCoord& coord) {
    return std::pow(normSquared(coord), 0.5);
}

template <std::ranges::random_access_range RangeCoordA, std::ranges::random_access_range RangeCoordB>
requires std::convertible_to<std::ranges::range_value_t<RangeCoordB>, std::ranges::range_value_t<RangeCoordA>>
    && std::convertible_to<std::ranges::range_value_t<RangeCoordA>, double>
double norm(const RangeCoordA& coord_a, const RangeCoordB& coord_b) {
    return std::pow(normSquared(coord_a, coord_b));
}

template <std::ranges::random_access_range RangeCoordRet, std::ranges::random_access_range RangeCoordA, std::ranges::random_access_range RangeCoordB>
requires std::convertible_to<std::ranges::range_value_t<RangeCoordB>, std::ranges::range_value_t<RangeCoordA>>
    && std::convertible_to<std::ranges::range_value_t<RangeCoordA>, std::ranges::range_value_t<RangeCoordRet>>
void sub(RangeCoordRet& coord_ret, const RangeCoordA& coord_a, const RangeCoordB& coord_b) {
    for (size_t i = 0; i < std::ranges::size(coord_ret); i++) {
        std::ranges::range_value_t<RangeCoordA> xa{};
        std::ranges::range_value_t<RangeCoordB> xb{};
        if (i < std::ranges::size(coord_a)) {
            xa = coord_a[i];
        }
        if (i < std::ranges::size(coord_b)) {
            xb = coord_b[i];
        }
        coord_ret[i] = static_cast<std::ranges::range_value_t<RangeCoordRet>>(xa - xb);
    }
}

template <std::ranges::random_access_range RangeCoordRet, std::ranges::random_access_range RangeCoordA, std::ranges::random_access_range RangeCoordB>
requires std::convertible_to<std::ranges::range_value_t<RangeCoordB>, std::ranges::range_value_t<RangeCoordA>>
    && std::convertible_to<std::ranges::range_value_t<RangeCoordA>, std::ranges::range_value_t<RangeCoordRet>>
void add(RangeCoordRet& coord_ret, const RangeCoordA& coord_a, const RangeCoordB& coord_b) {
    for (size_t i = 0; i < std::ranges::size(coord_ret); i++) {
        std::ranges::range_value_t<RangeCoordA> xa{};
        std::ranges::range_value_t<RangeCoordB> xb{};
        if (i < std::ranges::size(coord_a)) {
            xa = coord_a[i];
        }
        if (i < std::ranges::size(coord_b)) {
            xb = coord_b[i];
        }
        coord_ret[i] = static_cast<std::ranges::range_value_t<RangeCoordRet>>(xa + xb);
    }
}

template <typename T, std::ranges::random_access_range RangeCoordRet, std::ranges::random_access_range RangeCoordA>
requires std::convertible_to<std::ranges::range_value_t<RangeCoordA>, T>
    && std::convertible_to<std::ranges::range_value_t<RangeCoordA>, std::ranges::range_value_t<RangeCoordRet>>
void scale(RangeCoordRet& coord_ret, const RangeCoordA& coord_a, T factor) {
    for (size_t i = 0; i < std::ranges::size(coord_ret); i++) {
        std::ranges::range_value_t<RangeCoordA> xa{};
        if (i < std::ranges::size(coord_a)) {
            xa = coord_a[i];
        }
        coord_ret[i] = static_cast<std::ranges::range_value_t<RangeCoordRet>>(factor * xa);
    }
}

template <typename T, std::ranges::random_access_range RangeCoordA, std::ranges::random_access_range RangeCoordB>
requires std::convertible_to<std::ranges::range_value_t<RangeCoordB>, std::ranges::range_value_t<RangeCoordA>>
    && std::convertible_to<std::ranges::range_value_t<RangeCoordA>, T>
T dot(const RangeCoordA& coord_a, const RangeCoordB& coord_b) {
    T tot{};
    for (size_t i = 0; i < std::max(std::ranges::size(coord_a), std::ranges::size(coord_b)); i++) {
        std::ranges::range_value_t<RangeCoordA> xa{};
        std::ranges::range_value_t<RangeCoordB> xb{};
        if (i < std::ranges::size(coord_a)) {
            xa = coord_a[i];
        }
        if (i < std::ranges::size(coord_b)) {
            xb = coord_b[i];
        }
        tot += xa * xb;
    }
    return tot;
}

} // namespace skx::space::euc

namespace skx::space {

template <std::ranges::random_access_range RangeDims>
std::ranges::range_value_t<RangeDims> getVolume(const RangeDims& dims) {
    std::ranges::range_value_t<RangeDims> mult = 1;
    for (const std::ranges::range_value_t<RangeDims>& d : dims) {
        mult *= d;
    }
    return mult;
}    

template <std::ranges::random_access_range RangeDims, std::ranges::random_access_range RangeSpace, std::ranges::random_access_range RangeCoord>
    requires std::convertible_to<std::ranges::range_value_t<RangeDims>, size_t>
        && std::convertible_to<std::ranges::range_value_t<RangeSpace>, double>
        && std::convertible_to<std::ranges::range_value_t<RangeCoord>, size_t>
double& getFlatmapValueRefFromCoord(const RangeDims& dims, RangeSpace& space, const RangeCoord& coord) {
    static double inval = 0.0;
    if (std::ranges::size(dims) != std::ranges::size(coord)
        || std::ranges::size(dims) < 1
    ) {
        inval = 0.0;
        return inval; // error
    }
    size_t idx = 0;
    size_t mult = 1;
    for (size_t i = 0; i < std::ranges::size(dims); i++) {
        if (coord[i] >= dims[i]) {
            inval = 0.0;
            return inval;
        }
        idx += mult * coord[i];
        mult *= dims[i];
    }
    return space[idx];
}


template <std::ranges::random_access_range RangeDims, std::ranges::random_access_range RangeCoordsFlatmap>
std::span<std::ranges::range_value_t<RangeCoordsFlatmap>> getCoordRefFromCoordFlatmapIndex(const RangeDims& dims, RangeCoordsFlatmap& coords_flatmap, size_t idx) {
    return std::span<std::ranges::range_value_t<RangeCoordsFlatmap>>(&coords_flatmap[idx * std::ranges::size(dims)], std::ranges::size(dims));
}

template <typename T>
using PointView = std::span<T>;

template <typename T>
class PointCloud {
public:
    PointCloud(size_t ndims, size_t capacity_estimate) 
        : m_ndims(ndims), m_points_flatmap(), m_ndims_constrained_stack()
    {
        m_ndims_constrained_stack.push(get_ndims());
        m_points_flatmap.reserve(get_ndims()*capacity());
    }
    PointCloud() : PointCloud(1,0) {}

    size_t capacity() const {
        return m_points_flatmap.capacity() / m_ndims;
    }

    size_t size() const {
        return m_points_flatmap.size() / m_ndims;
    }

    void constrain_view(size_t ndims) const {
        if (ndims > get_ndims()) {
            ndims = get_ndims();
        }
        m_ndims_constrained_stack.push(ndims);
    }

    void unconstrain_view() const {
        if (false == m_ndims_constrained_stack.empty()) {
            m_ndims_constrained_stack.pop();
        }
    }

    PointView<T> at(size_t pos) {
        return PointView<T>(&m_points_flatmap.at(pos*m_ndims), get_ndims_constrained());
    }

    PointView<const T> at(size_t pos) const {
        return PointView<const T>(&m_points_flatmap.at(pos*m_ndims), get_ndims_constrained());
    }

    PointView<T> operator[](size_t pos) {
        return PointView<T>(&m_points_flatmap[pos*m_ndims], get_ndims_constrained());
    }

    PointView<const T> operator[](size_t pos) const {
        return PointView<const T>(&m_points_flatmap[pos*m_ndims], get_ndims_constrained());
    }

    template <std::ranges::input_range RangePoint> 
    requires std::convertible_to<std::ranges::range_value_t<RangePoint>, T>
    void push_back(RangePoint&& point) {
        for (const std::ranges::range_value_t<RangePoint>& e : point) {
            m_points_flatmap.push_back(static_cast<T>(e));
        }
    }

    size_t get_ndims() const {
        return m_ndims;
    }

    size_t get_ndims_constrained() const {
        return m_ndims_constrained_stack.top();
    }

private:
    size_t m_ndims;
    mutable std::stack<size_t> m_ndims_constrained_stack;
    std::vector<T> m_points_flatmap;
};

enum class DistanceMetric {
    EUCLIDEAN,
};

enum class PointCloudOrder {
    MOST_SIG_LAST_AXIS, // 3ord: 000, 100, 200, 010, 110, 210, 020, ..., 001, 101, ...
};

template<typename T, std::ranges::random_access_range RangeDims>
PointCloud<T> createOrderedUniformPointCloud(PointCloudOrder order, const RangeDims& dims) {
    PointCloud<T> point_cloud(dims.size(), getVolume(dims));
    switch (order)
    {
    case PointCloudOrder::MOST_SIG_LAST_AXIS:
        {
            std::vector<size_t> X(dims.size());
            bool exausted = false;
            while (false == exausted) {
                point_cloud.push_back(X);
                X[0]++;
                size_t dim_ix = 0;
                while (X[dim_ix] == dims[dim_ix]) {
                    X[dim_ix] = 0;
                    dim_ix++;
                    if (dim_ix == dims.size()) {
                        exausted = true;
                        break;
                    }
                    X[dim_ix]++;
                }
            }
        }
        break;    
    default:
        break;
    }
    return point_cloud;
}

template<typename T, std::ranges::random_access_range RangePoint, std::ranges::random_access_range RangeDims>
ssize_t getOrderedUniformPointCloudKthMatchIndex(PointCloudOrder order, const RangeDims& dims, const PointCloud<T>& point_cloud, const RangePoint& point, size_t kth_match) { 
    switch (order) {
    case PointCloudOrder::MOST_SIG_LAST_AXIS:
        {
            size_t idx = 0;
            size_t mult = 1;
            for (size_t i = 0; i < std::min(dims.size(),point.size()); i++) {
                idx += mult * point[i];
                mult *= dims[i];
            }
            idx *= kth_match;
            if (idx > point_cloud.size()) {
                return -1;
            }
            return idx;
        }
        break;
    default:
        break;
    }
    return -1;
}

template <typename T>
std::string pv_to_string(const PointView<T>& pv) {
    std::string st;
    st.reserve(pv.size() * 7);
    st += "( ";
    for (const T& e : pv) {
        st += std::to_string(e) + " ";
    }
    st += ")";
    return st;
}

template <typename T>
struct BallTree {
    struct BallTreeNode {
        size_t centroid_index;
        size_t points_offset_index;
        size_t num_elements;
        ssize_t child_indices[2]; 
        double radius2 = 0.0;
    };
    struct KNearRefPoint {
        size_t point_index;
        T dis2_q_to_p;
    };
    struct KNearRefPointCompare {
        bool operator()(const KNearRefPoint& rpa, const KNearRefPoint& rpb) {
            return rpa.dis2_q_to_p < rpb.dis2_q_to_p;
        }
    };
    using KNearestMaxHeap = std::priority_queue<typename BallTree<T>::KNearRefPoint,std::vector<typename BallTree<T>::KNearRefPoint>,typename BallTree<T>::KNearRefPointCompare>;
    std::vector<BallTreeNode> nodes;
    PointCloud<T> pcc;
    std::vector<size_t> indices;
    size_t ndims;
};

template <typename T>
BallTree<T>& balltreeCreate(BallTree<T>& balltree, const PointCloud<T>& point_cloud, size_t leaf_size, DistanceMetric distance_metric, size_t ndims_constraint) {

    if (leaf_size < 3) {
        leaf_size = 3;
    }
    if (ndims_constraint > point_cloud.get_ndims()) {
        ndims_constraint = point_cloud.get_ndims();
    }
    balltree.ndims = ndims_constraint;
    size_t& ndims = balltree.ndims;

    point_cloud.constrain_view(ndims);

    balltree.pcc = PointCloud<T>(ndims, 2 * static_cast<size_t>(std::ceil(static_cast<double>(point_cloud.size())/static_cast<double>(leaf_size))) - 1);
    PointCloud<T>& pcc = balltree.pcc;

    std::vector<typename BallTree<T>::BallTreeNode>& nodes = balltree.nodes;

    balltree.indices = std::vector<size_t>(point_cloud.size(), 0);
    std::vector<size_t>& indices = balltree.indices;
    for (size_t i = 0; i < indices.size(); i++) {
        indices[i] = i;
    }
    std::vector<size_t> indices_local(indices.size(), 0);

    std::vector<double> prscs(indices.size(), 0);

    std::vector<T> point_tmp(ndims);
    std::vector<T> point_mean(ndims);
    std::vector<T> aop(ndims);

    T dis2_max = 0;

    struct NodeBuilder {
        std::vector<size_t>::iterator beg;
        std::vector<size_t>::iterator end;
        ssize_t parent_index;
        size_t parent_child_index;
    };
    std::stack<NodeBuilder> subranges_stack;
    subranges_stack.push(NodeBuilder{indices.begin(),indices.end(),-1});

    while (false == subranges_stack.empty()) {
        NodeBuilder subrange = subranges_stack.top();
        subranges_stack.pop();
        std::vector<size_t>::iterator& beg = subrange.beg;
        std::vector<size_t>::iterator& end = subrange.end;

        nodes.push_back(typename BallTree<T>::BallTreeNode());
        size_t node_index = nodes.size()-1;
        typename BallTree<T>::BallTreeNode& node = nodes[node_index];
        if (-1 != subrange.parent_index) {
            nodes[subrange.parent_index].child_indices[subrange.parent_child_index] = node_index;
        }
        const size_t n = static_cast<const size_t>(std::distance(beg, end));
        node.num_elements = n;
        node.points_offset_index = static_cast<size_t>(std::distance(indices.begin(), beg));
        for (size_t i = 0; i < sizeof(node.child_indices)/sizeof(node.child_indices[0]); i++) {
            node.child_indices[i] = {-1};
        }
        skx::space::euc::scale(point_mean, point_tmp, 0.0);
        for (const size_t& idx : std::ranges::subrange(beg,end)) {
            skx::space::euc::scale(point_tmp, point_cloud[idx], 1.0/static_cast<double>(n));
            skx::space::euc::add(point_mean, point_mean, point_tmp);
        }
        pcc.push_back(point_mean);
        node.centroid_index = pcc.size()-1;
        node.radius2 = 0;
        for (const size_t& idx : std::ranges::subrange(beg,end)) {
            skx::space::euc::sub(point_tmp, point_mean, point_cloud[idx]);
            double dis2 = skx::space::euc::normSquared(point_tmp);
            if (dis2 > node.radius2) {
                node.radius2 = dis2;
            }
        }

        if (n < leaf_size) {
            continue;
        }

        indices_local.resize(n);
        for (size_t i = 0; i < indices_local.size(); i++) {
            indices_local[i] = i;
        }

        PointView<const T> popA = point_cloud[*beg];
        PointView<const T> popB = point_cloud[*beg];

        for (size_t jump_count = 0; jump_count < 2; jump_count++) {
            dis2_max = 0;
            for (const size_t& idx : std::ranges::subrange(beg,end)) {
                skx::space::euc::sub(point_tmp, popA, point_cloud[idx]);
                double dis2 = skx::space::euc::normSquared(point_tmp);
                if (dis2 > dis2_max) {
                    popB = point_cloud[idx];
                    dis2_max = dis2;
                }
            }
            if (0 == jump_count) {
                popA = popB;
                dis2_max = 0;
            }
        }

        skx::space::euc::sub(aop, popB, popA);
        double aopdotpop = skx::space::euc::dot<double>(popA, aop);
        double aopdotaop = skx::space::euc::normSquared(aop);

        prscs.resize(indices_local.size());
        size_t prscs_i = 0;
        for (const size_t& idx : std::ranges::subrange(beg,end)) { 
            double projscnum = (skx::space::euc::dot<double>(point_cloud[idx], aop) - aopdotpop);// / aopdotaop;
            prscs[prscs_i] = projscnum;
            prscs_i++;
        }
        size_t split_offset = indices_local.size()/2;
        std::nth_element(indices_local.begin(), indices_local.begin() + split_offset, indices_local.end(), [&prscs](const size_t& a, const size_t& b)->bool{
            return prscs[a] < prscs[b];
        });
        for (size_t i = 0; i < indices_local.size(); i++) {
            indices_local[i] = *(beg + indices_local[i]);
        } 
        size_t copyback_i = 0;
        for (size_t& idx : std::ranges::subrange(beg,end)) {
            idx = indices_local[copyback_i];
            copyback_i++;
        }
        
        NodeBuilder builder;

        builder.beg = beg;
        builder.end = beg+split_offset;
        builder.parent_index = node_index;
        builder.parent_child_index = 0;
        subranges_stack.push(builder);

        builder.beg = beg+split_offset;
        builder.end = end;
        builder.parent_index = node_index;
        builder.parent_child_index = 1;
        subranges_stack.push(builder);

    }

    // size_t node_i = 0;
    // for (const BallTreeNode& node : nodes) {
    //     std::cout << "NODE " << node_i << "\n"
    //         << "left " << node.child_indices[0] << "\n"
    //         << "rght " << node.child_indices[1] << "\n"
    //         << "ofst " << node.points_offset_index << "\n"
    //         << "nume " << node.num_elements << "\n"
    //         << "cent " << pv_to_string(pcc[node.centroid_index]) << "\n"
    //         << "rad2 " << node.radius2 << "\n"
    //         << "are  ";
    //         for (size_t i = 0; i < node.num_elements; i++) {
    //             std::cout << pv_to_string(point_cloud[indices[node.points_offset_index+i]]) << " ";
    //         }
    //         std::cout << "\n";
    //         node_i++;
    // }

    point_cloud.unconstrain_view();

    return balltree;
}

template <typename T>
typename BallTree<T>::KNearestMaxHeap balltreeKNearest(const BallTree<T>& balltree, const PointCloud<T>& point_cloud, const PointView<const T>& q, size_t k)
{

    const PointCloud<T>& pcc = balltree.pcc;
    const std::vector<typename BallTree<T>::BallTreeNode>& nodes = balltree.nodes;
    const std::vector<size_t>& indices = balltree.indices;
    const size_t& ndims = balltree.ndims;

    point_cloud.constrain_view(ndims);

    std::priority_queue<typename BallTree<T>::KNearRefPoint,std::vector<typename BallTree<T>::KNearRefPoint>,typename BallTree<T>::KNearRefPointCompare> knear;
    std::vector<T> point_tmp(ndims);
    std::stack<size_t> nodeidx_stack;
    nodeidx_stack.push(0);
    while (false == nodeidx_stack.empty()) {
        const typename BallTree<T>::BallTreeNode& node = nodes[nodeidx_stack.top()];
        nodeidx_stack.pop();
        // std::cout << "check node: " << pv_to_string(pcc[node.centroid_index]) << "\n";
        
        skx::space::euc::sub(point_tmp, q, pcc[node.centroid_index]);
        double dis2_to_centroid = skx::space::euc::normSquared(point_tmp);
        if (knear.size() >= k && std::pow(knear.top().dis2_q_to_p,0.5) < std::pow(dis2_to_centroid,0.5) - std::pow(node.radius2,0.5)) {
            // std::cout << "(skipping)\n";
            continue;
        }
        if (-1 == node.child_indices[0]) { // leaf node
            // std::cout << "(leaf)\n";
            double dis2_q_to_p = INFINITY;
            for (size_t i = node.points_offset_index; i < node.points_offset_index + node.num_elements; i++) {
                const PointView<const T>& pv = point_cloud[indices[i]];
                skx::space::euc::sub(point_tmp, q, pv);
                dis2_q_to_p = skx::space::euc::normSquared(point_tmp);
                if (knear.size() > 0 && knear.size() >= k && dis2_q_to_p > knear.top().dis2_q_to_p) {
                    // std::cout << ".";
                    continue;
                }
                else if (knear.size() >= k) {
                    // std::cout << "(-)";
                    knear.pop();
                }
                // std::cout << "+";
                typename BallTree<T>::KNearRefPoint rp;
                rp.dis2_q_to_p = dis2_q_to_p;
                rp.point_index = indices[i];
                knear.push(rp);
            }
            // std::cout << "\n";
            continue;
        }
        size_t child_node_closest_idx = 0;
        double dis2_min = INFINITY;
        size_t dis2idx_min = 0;
        for (size_t child_i = 0; child_i < sizeof(node.child_indices)/sizeof(node.child_indices[0]); child_i++) {
            skx::space::euc::sub(point_tmp, q, pcc[node.centroid_index]);
            double dis2 = skx::space::euc::normSquared(point_tmp);
            if (dis2 <= node.radius2) {
                child_node_closest_idx = child_i;
            }
        }
        nodeidx_stack.push(node.child_indices[(child_node_closest_idx+1)%2]);
        nodeidx_stack.push(node.child_indices[child_node_closest_idx]);
    }
    // std::cout << "knear ";
    // while (false == knear.empty())
    // {
    //     std::cout << pv_to_string(point_cloud[knear.top().point_index]) << " ";
    //     knear.pop();
    // }
    // std::cout << "\n";
    return knear;
}


#if 0
        std::cout << "( ";
        for (const size_t& idx : indices) {
            if (&idx == &*beg || &idx == &*(end-1)) {
                std::cout << "*";
            }
            std::cout << idx << " ";
        }
        std::cout << ").\n";
        std::cout << "( ";
        for (size_t i = 0; i < point_cloud.size(); i++) {
            std::cout << pv_to_string(point_cloud[indices[i]]) << " ";
        }
        std::cout << ").\n";

        if (n < leaf_size) {
            std::cout << "STOP\n";
            continue;
        }

        for (size_t i = 0; i < indices_local.size(); i++) {
            std::string st;
            st += "[";
            st += std::to_string(i);
            st += "]";
            st += "->";
            st += std::to_string(indices_local[i]);
            st += "->";
            st += pv_to_string(point_cloud[indices_local[i]]);
            st += "\n";
            skx::io::log(skx::io::getDefaultLoggerRef(), st);
        }
    std::cout << "-----\n";
    {
        std::string st;
        st = pv_to_string(point_ref);
        skx::io::log(skx::io::getDefaultLoggerRef(), st);
        st = pv_to_string(point_furthest);
        skx::io::log(skx::io::getDefaultLoggerRef(), st + "\n");
    }
        {
            std::string st;
            st += pv_to_string(point_cloud[i]);
            st += " -> " + std::to_string(projscnum);
            skx::io::log(skx::io::getDefaultLoggerRef(), st + "\n");
        }
    std::cout << "mid = " << indices.size() / 2 << "\n";
    std::cout << "-----\n";
    for (size_t i = 0; i < point_cloud.size(); i++) {
        {
            std::string st;
            st += std::to_string(indices[i]) + " ";
            st += pv_to_string(point_cloud[indices[i]]);
            st += " -> " + std::to_string(prscs[indices[i]]);
            skx::io::log(skx::io::getDefaultLoggerRef(), st + "\n");
        }
    }
#endif

} // namespace skx::space

