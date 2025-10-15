#if defined(RBF) || defined(TESTING)
#include "RBF.hpp"
#include <random>
#include <ctime>

// ------------------------------ ctor ------------------------------
RBFModel::RBFModel(int dimensions, int dimensionSize, int totalQueries)
    : Model(dimensions, dimensionSize, totalQueries),
      stateSpace(new KDTreeStateSpace(dimensions, dimensionSize)) {
    std::srand((unsigned)std::time(nullptr));
}

// ------------------------------ key helper ------------------------------
std::string RBFModel::coords_key(const std::vector<int>& c) {
    std::ostringstream os;
    for (size_t i = 0; i < c.size(); ++i) {
        if (i) os << ',';
        os << c[i];
    }
    return os.str();
}

// ------------------------------ sampling policy ------------------------------
std::vector<int> RBFModel::get_next_query() {
    currentQuery++;
    return is_dense_sampling() ? dense_uniform_query() : farthest_point_query();
}

std::vector<int> RBFModel::dense_uniform_query() {
    int D = dimensions, K = dimensionSize;
    std::vector<int> q(D, 0);
    double step = (K - 1.0) / std::max(1, totalQueries);
    for (int d = 0; d < D; ++d) {
        double phase = 0.5 * (d + 1);
        double val = step * (currentQuery - 1 + phase);
        int idx = (int)std::round(std::fmod(std::max(0.0, val), (double)(K - 1)));
        q[d] = std::min(std::max(idx, 0), K - 1);
    }
    return q;
}

std::vector<int> RBFModel::farthest_point_query() {
    int D = dimensions, K = dimensionSize;
    std::vector<int> best(D, 0);
    double bestDist = -1.0;
    if (sample_coords.empty()) {
        for (int d = 0; d < D; ++d) best[d] = K / 2;
        return best;
    }

    int stride = (K >= 128) ? 8 : (K >= 64 ? 4 : 1);
    for (int i0 = 0; i0 < K; i0 += stride) {
        if (D == 2) {
            for (int i1 = 0; i1 < K; i1 += stride) {
                std::vector<int> c = {i0, i1};
                auto* nn = stateSpace->nearest_neighbor(c);
                double d2 = stateSpace->squared_distance(c, nn->coords);
                if (d2 > bestDist) { bestDist = d2; best = c; }
            }
        } else if (D == 3) {
            for (int i1 = 0; i1 < K; i1 += stride)
                for (int i2 = 0; i2 < K; i2 += stride) {
                    std::vector<int> c = {i0, i1, i2};
                    auto* nn = stateSpace->nearest_neighbor(c);
                    double d2 = stateSpace->squared_distance(c, nn->coords);
                    if (d2 > bestDist) { bestDist = d2; best = c; }
                }
        }
    }
    return best;
}

// ------------------------------ update & train ------------------------------
void RBFModel::update_prediction(const std::vector<int>& query, double result) {
    stateSpace->insert(query, result);
    std::string key = coords_key(query);

    if (seen_keys.insert(key).second) {
        sample_coords.push_back(query);
        sample_values.push_back(result);
    } else {
        for (size_t i = 0; i < sample_coords.size(); ++i)
            if (sample_coords[i] == query)
                sample_values[i] = result;
    }

    if (currentQuery == totalQueries) {
        select_kernel_and_params();
    }
}

double RBFModel::get_value_at(const std::vector<int>& query) {
    if (trained()) {
        double y = 0.0;
        for (size_t i = 0; i < sample_coords.size(); ++i)
            y += weights[i] * phi(euclid_scaled(query, sample_coords[i], dimensionSize));
        if (kernel == TPS && tps_affine.size() == (size_t)(dimensions + 1)) {
            y += tps_affine[0];
            for (int d = 0; d < dimensions; ++d)
                y += tps_affine[d + 1] * query[d];
        }
        return y;
    }
    double v = stateSpace->get(query);
    if (v != 0.0) return v;
    auto* nn = stateSpace->nearest_neighbor(query);
    return nn ? nn->value : 0.0;
}

// ------------------------------ kernel selection ------------------------------
void RBFModel::select_kernel_and_params() {
    const int D = dimensions;
    const int K = dimensionSize;
    double rho = budget_ratio();
    double dnn = median_1nn_distance(sample_coords, K);
    if (dnn <= 0.0) dnn = 1.0;

    // --- Adaptive kernel selection (robust) ---
    if (D == 2 && rho >= 0.05 && rho <= 0.3)
        kernel = MQ;           // MQ safer than TPS in competition // TODO: revisit - use TPS?
    else if (rho >= 0.25)
        kernel = GAUSSIAN;
    else if (rho >= 0.07)
        kernel = MQ;
    else
        kernel = IMQ;

    // --- Shape parameter (ε) ---
    epsilon = 0.3 / dnn;   // smoother than 1/dnn

    // --- Ridge term (λ) ---
    lambda = (kernel == TPS ? 1e-6 : 1e-8);

    // --- Compute weights with fallback ---
    try {
        compute_global_weights();
    } catch (const std::exception& e) {
        std::cerr << "[RBF] Kernel " << kernel
                  << " failed (" << e.what() << "). Falling back to MQ.\n";
        kernel = MQ;
        lambda = 1e-6;
        compute_global_weights();
    }
}

// ------------------------------ training ------------------------------
void RBFModel::compute_global_weights() {
    const int N = (int)sample_coords.size();
    const int K = dimensionSize;
    if (N == 0) return;

    std::vector<std::vector<double>> A(N, std::vector<double>(N, 0.0));
    for (int i = 0; i < N; ++i) {
        A[i][i] = phi(0.0) + lambda;
        for (int j = i + 1; j < N; ++j) {
            double r = euclid_scaled(sample_coords[i], sample_coords[j], K);
            double v = phi(r);
            A[i][j] = A[j][i] = v;
        }
    }

    std::vector<double> b = sample_values;
    std::vector<double> x(N, 0.0);
    if (!solve_linear_system(A, b, x))
        throw std::runtime_error("RBF: linear system solve failed (ill-conditioned).");
    weights = std::move(x);
}

// ------------------------------ kernels ------------------------------
double RBFModel::phi(double r) const {
    double er = epsilon * r;
    switch (kernel) {
        case GAUSSIAN: return std::exp(-(er * er));
        case MQ:       return std::sqrt(1.0 + (er * er));
        case IMQ:      return 1.0 / std::sqrt(1.0 + (er * er));
        case TPS:      return (r <= 0.0) ? 0.0 : (r * r) * std::log(r);
        default:       return 0.0;
    }
}

// ------------------------------ helpers ------------------------------
double RBFModel::euclid_scaled(const std::vector<int>& a, const std::vector<int>& b, int K) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double da = (double)a[i] / (K - 1);
        double db = (double)b[i] / (K - 1);
        double d = da - db;
        s += d * d;
    }
    return std::sqrt(s);
}

double RBFModel::median_1nn_distance(const std::vector<std::vector<int>>& X, int K) {
    const int N = (int)X.size();
    if (N <= 1) return 1.0;
    std::vector<double> nn(N, std::numeric_limits<double>::infinity());
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (i != j)
                nn[i] = std::min(nn[i], euclid_scaled(X[i], X[j], K));
    std::sort(nn.begin(), nn.end());
    return (N % 2) ? nn[N/2] : 0.5 * (nn[N/2 - 1] + nn[N/2]);
}

double RBFModel::pow_int(double base, int exp) const {
    double p = 1.0;
    for (int i = 0; i < exp; ++i) p *= base;
    return p;
}

double RBFModel::budget_ratio() const {
    const int N = (int)sample_coords.size();
    double denom = pow_int((double)dimensionSize, dimensions);
    return denom > 0.0 ? (double)N / denom : 0.0;
}

bool RBFModel::is_dense_sampling() const { return budget_ratio() >= 0.25; }

// ------------------------------ dense solver ------------------------------
bool RBFModel::solve_linear_system(std::vector<std::vector<double>>& A,
                                   std::vector<double>& b,
                                   std::vector<double>& x) {
    int n = (int)A.size();
    if (n == 0) return true;
    for (auto& row : A) if ((int)row.size() != n) return false;
    if ((int)b.size() != n) return false;

    for (int i = 0; i < n; ++i) A[i].push_back(b[i]);
    for (int col = 0; col < n; ++col) {
        int piv = col;
        double best = std::fabs(A[col][col]);
        for (int r = col + 1; r < n; ++r)
            if (std::fabs(A[r][col]) > best) { best = std::fabs(A[r][col]); piv = r; }
        if (best == 0.0) return false;
        if (piv != col) std::swap(A[piv], A[col]);
        double pivVal = A[col][col];
        for (int c = col; c <= n; ++c) A[col][c] /= pivVal;
        for (int r = col + 1; r < n; ++r) {
            double f = A[r][col];
            for (int c = col; c <= n; ++c) A[r][c] -= f * A[col][c];
        }
    }
    x.assign(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double sum = A[i][n];
        for (int c = i + 1; c < n; ++c) sum -= A[i][c] * x[c];
        x[i] = sum;
    }
    return true;
}
#endif
