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

// ------------------------------ sampling policy ------------------------------
std::vector<int> RBFModel::get_next_query() {
    currentQuery++;

    // If dense regime -> even coverage; else -> farthest-point
    if (is_dense_sampling()) {
        return dense_uniform_query();
    } else {
        return farthest_point_query();
    }
}

// Even coverage on a D-dim integer grid [0..K-1]^D
// Simple low-coupling scheme: each dim uses a different fractional phase.
std::vector<int> RBFModel::dense_uniform_query() {
    int D = dimensions;
    int K = dimensionSize;
    std::vector<int> q(D, 0);

    // Spread currentQuery across [0, K-1] with dim-dependent phases
    double step = (K - 1.0) / std::max(1, totalQueries);
    for (int d = 0; d < D; ++d) {
        double phase = 0.5 * (d + 1); // simple stagger
        double val = step * (currentQuery - 1 + phase);
        int idx = (int)std::round(std::fmod(std::max(0.0, val), (double)(K - 1)));
        q[d] = std::min(std::max(idx, 0), K - 1);
    }
    return q;
}

// Farthest-point sampling on the discrete grid using KD-tree nearest distance.
std::vector<int> RBFModel::farthest_point_query() {
    int D = dimensions;
    int K = dimensionSize;
    std::vector<int> bestCoord(D, 0);
    double bestDist = -1.0;

    // If no samples yet, choose middle of the grid
    if (sample_coords.empty()) {
        for (int d = 0; d < D; ++d) bestCoord[d] = K / 2;
        return bestCoord;
    }

    // Coarse scan: stride to keep cost down in large grids.
    int stride = 1;
    if (K >= 64) stride = 4;
    if (K >= 128) stride = 8;

    // Enumerate a coarse lattice and pick the point that maximizes distance to nearest sample.
    // For D<=3 this is fine; for larger D we’d switch to random candidates.
    for (int i0 = 0; i0 < K; i0 += stride) {
        if (D == 1) {
            std::vector<int> c = { i0 };
            // nearest distance via KD-tree
            auto* nn = stateSpace->nearest_neighbor(c);
            double d2 = stateSpace->squared_distance(c, nn->coords);
            if (d2 > bestDist) { bestDist = d2; bestCoord = c; }
        } else if (D == 2) {
            for (int i1 = 0; i1 < K; i1 += stride) {
                std::vector<int> c = { i0, i1 };
                auto* nn = stateSpace->nearest_neighbor(c);
                double d2 = stateSpace->squared_distance(c, nn->coords);
                if (d2 > bestDist) { bestDist = d2; bestCoord = c; }
            }
        } else if (D == 3) {
            for (int i1 = 0; i1 < K; i1 += stride) {
                for (int i2 = 0; i2 < K; i2 += stride) {
                    std::vector<int> c = { i0, i1, i2 };
                    auto* nn = stateSpace->nearest_neighbor(c);
                    double d2 = stateSpace->squared_distance(c, nn->coords);
                    if (d2 > bestDist) { bestDist = d2; bestCoord = c; }
                }
            }
        }
    }
    return bestCoord;
}

// ------------------------------ updates & training ------------------------------
void RBFModel::update_prediction(const std::vector<int>& query, double result) {
    // Insert into KD-tree and local buffers
    stateSpace->insert(query, result);
    sample_coords.push_back(query);
    sample_values.push_back(result);

    // When last query arrives -> select kernel & params, then train global RBF
    if (currentQuery == totalQueries) {
        select_kernel_and_params();
        compute_global_weights();
    }
}

double RBFModel::get_value_at(const std::vector<int>& query) {
    // If we have trained weights, do full global prediction
    if (trained()) {
        double y = 0.0;
        for (size_t i = 0; i < sample_coords.size(); ++i) {
            y += weights[i] * phi(euclid(query, sample_coords[i]));
        }
        return y;
    }

    // If not trained yet:
    // 1) If exact sample exists, return it (KD-tree get returns 0.0 if missing)
    double v = stateSpace->get(query);
    if (v != 0.0) return v;

    // 2) Otherwise, nearest-neighbor fallback
    auto* nn = stateSpace->nearest_neighbor(query);
    return nn ? nn->value : 0.0;
}

// ------------------------------ kernel selection ------------------------------
void RBFModel::select_kernel_and_params() {
    // Heuristics:
    // - rho = N / K^D
    // - d_nn = median 1-NN among observed samples
    // - Rules from design doc (Gaussian for dense & smooth; MQ for medium; IMQ for sparse/uneven; TPS for 2D/3D surface need)

    const int D = dimensions;
    const int K = dimensionSize;
    const int N = (int)sample_coords.size();

    double rho = budget_ratio(); // N / K^D
    double dnn = median_1nn_distance(sample_coords);
    if (dnn <= 0.0) dnn = 1.0;

    // Choose kernel
    if ((D == 2 || D == 3) && rho >= 0.02 && rho <= 0.3) {
        // 2D/3D smooth “surface” regime
        kernel = TPS;
    } else if (rho >= 0.25) {
        kernel = GAUSSIAN;
    } else if (rho >= 0.07) {
        kernel = MQ;
    } else {
        kernel = IMQ;
    }

    // Set epsilon from d_nn
    // epsilon ≈ c / d_nn with c in [0.5, 2]. Start with c=1.0.
    epsilon = 1.0 / dnn;

    // Tiny ridge to stabilize if needed (only for GAUSSIAN/MQ/IMQ; keep TPS exact)
    lambda = (kernel == TPS ? 0.0 : 1e-10);
}

// ------------------------------ training (global) ------------------------------
void RBFModel::compute_global_weights() {
    const int N = (int)sample_coords.size();
    if (N == 0) return;

    // Build Phi
    std::vector<std::vector<double>> A(N, std::vector<double>(N, 0.0));
    for (int i = 0; i < N; ++i) {
        A[i][i] = phi(0.0) + lambda; // diagonal plus ridge
        for (int j = i + 1; j < N; ++j) {
            double r = euclid(sample_coords[i], sample_coords[j]);
            double v = phi(r);
            A[i][j] = v;
            A[j][i] = v;
        }
    }

    // Right-hand side
    std::vector<double> b = sample_values;
    std::vector<double> x(N, 0.0);

    bool ok = solve_linear_system(A, b, x);
    if (!ok) {
        // If failed due to conditioning, try a slightly larger ridge and re-solve once
        if (lambda == 0.0) lambda = 1e-8; else lambda *= 10.0;

        for (int i = 0; i < N; ++i) {
            for (int j = i; j < N; ++j) {
                double r = (i == j) ? 0.0 : euclid(sample_coords[i], sample_coords[j]);
                double v = phi(r);
                A[i][j] = v + ((i == j) ? lambda : 0.0);
                A[j][i] = A[i][j];
            }
        }
        x.assign(N, 0.0);
        ok = solve_linear_system(A, b, x);
        if (!ok) throw std::runtime_error("RBF: linear system solve failed (ill-conditioned).");
    }
    weights = std::move(x);
}

// ------------------------------ kernels ------------------------------
double RBFModel::phi(double r) const {
    // Handle r=0 case for TPS: phi(0) = 0 (limit)
    if (kernel == TPS) {
        if (r <= 0.0) return 0.0;
        return (r * r) * std::log(r);
    }
    // Gaussian / MQ / IMQ
    const double er = epsilon * r;
    switch (kernel) {
        case GAUSSIAN: return std::exp(-(er * er));
        case MQ:       return std::sqrt(1.0 + (er * er));
        case IMQ:      return 1.0 / std::sqrt(1.0 + (er * er));
        default:       return 0.0;
    }
}

// ------------------------------ helpers ------------------------------
double RBFModel::euclid(const std::vector<int>& a, const std::vector<int>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = double(a[i] - b[i]);
        s += d * d;
    }
    return std::sqrt(s);
}

// Median 1-NN distance among observed samples (O(N^2), fine for N<=200)
double RBFModel::median_1nn_distance(const std::vector<std::vector<int>>& X) {
    const int N = (int)X.size();
    if (N <= 1) return 1.0;
    std::vector<double> nn(N, std::numeric_limits<double>::infinity());
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) if (i != j) {
            double d = euclid(X[i], X[j]);
            if (d < nn[i]) nn[i] = d;
        }
    }
    std::sort(nn.begin(), nn.end());
    return (N % 2) ? nn[N/2] : 0.5 * (nn[N/2 - 1] + nn[N/2]);
}

double RBFModel::pow_int(double base, int exp) const {
    double p = 1.0;
    for (int i = 0; i < exp; ++i) p *= base;
    return p;
}

double RBFModel::budget_ratio() const {
    // rho = N / K^D
    const int N = (int)sample_coords.size();
    double denom = pow_int((double)dimensionSize, dimensions);
    if (denom <= 0.0) return 0.0;
    return (double)N / denom;
}

bool RBFModel::is_dense_sampling() const {
    // Simple threshold on rho. Tune as needed.
    return budget_ratio() >= 0.25;
}

// ------------------------------ dense solver ------------------------------
// Gaussian elimination with partial pivoting
bool RBFModel::solve_linear_system(std::vector<std::vector<double>>& A,
                                   std::vector<double>& b,
                                   std::vector<double>& x) {
    const int n = (int)A.size();
    if (n == 0) return true;
    for (auto& row : A) if ((int)row.size() != n) return false;
    if ((int)b.size() != n) return false;

    // Augment A | b
    for (int i = 0; i < n; ++i) A[i].push_back(b[i]);

    // Forward elimination with pivoting
    for (int col = 0; col < n; ++col) {
        // Pivot
        int piv = col;
        double best = std::fabs(A[col][col]);
        for (int r = col + 1; r < n; ++r) {
            double v = std::fabs(A[r][col]);
            if (v > best) { best = v; piv = r; }
        }
        if (best == 0.0) return false;
        if (piv != col) std::swap(A[piv], A[col]);

        // Normalize pivot row
        double pivVal = A[col][col];
        for (int c = col; c <= n; ++c) A[col][c] /= pivVal;

        // Eliminate below
        for (int r = col + 1; r < n; ++r) {
            double factor = A[r][col];
            if (factor == 0.0) continue;
            for (int c = col; c <= n; ++c) A[r][c] -= factor * A[col][c];
        }
    }

    // Back substitution
    x.assign(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double sum = A[i][n];
        for (int c = i + 1; c < n; ++c) sum -= A[i][c] * x[c];
        x[i] = sum; // A[i][i] should be 1 after normalization
    }
    return true;
}

#endif // RBF || TESTING
