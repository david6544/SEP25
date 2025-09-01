#include "MLP.hpp"
#include <random>
#include <cmath>
#include <cassert>

static std::mt19937 rng(12345);

static double rand_uniform(double a = -1.0, double b = 1.0) {
    std::uniform_real_distribution<double> dist(a, b);
    return dist(rng);
}

// -------------------- Activations --------------------
Activation sigmoid = {
    [](double x){ return 1.0 / (1.0 + std::exp(-x)); },
    [](double x){ double s = 1.0 / (1.0 + std::exp(-x)); return s*(1-s); }
};

Activation relu = {
    [](double x){ return x > 0.0 ? x : 0.0; },
    [](double x){ return x > 0.0 ? 1.0 : 0.0; }
};

Activation tanh_act = {
    [](double x){ return std::tanh(x); },
    [](double x){ double t = std::tanh(x); return 1 - t*t; }
};

Activation identity = {
    [](double x){ return x; },
    [](double){ return 1.0; }
};

// -------------------- Layer --------------------
Layer::Layer(int in_, int out_, Activation act_)
    : in(in_), out(out_), act(act_),
      W(Mat(out_, Vec(in_))), b(Vec(out_)),
      z(Vec(out_)), a(Vec(out_))
{
    double limit = std::sqrt(6.0 / (in + out));
    for (int i=0;i<out;i++){
        for (int j=0;j<in;j++){
            W[i][j] = rand_uniform(-limit, limit);
        }
        b[i] = 0.0;
    }
}

Vec Layer::forward(const Vec& x) {
    assert((int)x.size() == in);
    for (int i=0;i<out;i++){
        double sum = b[i];
        for (int j=0;j<in;j++) sum += W[i][j] * x[j];
        z[i] = sum;
        a[i] = act.f(sum);
    }
    return a;
}

// -------------------- MLP --------------------
MLP::MLP(const std::vector<int>& sizes,
         const std::vector<Activation>& acts)
{
    assert(sizes.size() >= 2);
    assert(acts.size() == sizes.size() - 1);
    for (size_t i=0;i+1<sizes.size();++i){
        layers.emplace_back(sizes[i], sizes[i+1], acts[i]);
    }
}

Vec MLP::predict(const Vec& x) {
    Vec cur = x;
    for (auto& L : layers) cur = L.forward(cur);
    return cur;
}

double MLP::train_sample(const Vec& x, const Vec& y, double lr) {
    // forward
    Vec cur = x;
    for (auto& L : layers) cur = L.forward(cur);
    Vec out = cur;

    assert(out.size() == y.size());

    // output delta
    int Lidx = (int)layers.size() - 1;
    Vec delta(layers[Lidx].out);
    for (int i=0;i<layers[Lidx].out;i++){
        double err = out[i] - y[i];
        delta[i] = 2.0 * err * layers[Lidx].act.df(layers[Lidx].z[i]);
    }

    // backprop
    for (int li=Lidx; li>=0; --li) {
        Layer& L = layers[li];
        Vec prev_a = (li==0) ? x : layers[li-1].a;

        // update weights/biases
        for (int i=0;i<L.out;i++){
            for (int j=0;j<L.in;j++){
                double grad_w = delta[i] * prev_a[j];
                L.W[i][j] -= lr * grad_w;
            }
            L.b[i] -= lr * delta[i];
        }

        if (li > 0) {
            Vec prev_delta(layers[li-1].out, 0.0);
            for (int pj=0; pj<layers[li-1].out; ++pj) {
                double accum = 0.0;
                for (int i=0;i<L.out;i++) {
                    accum += L.W[i][pj] * delta[i];
                }
                prev_delta[pj] = accum * layers[li-1].act.df(layers[li-1].z[pj]);
            }
            delta = std::move(prev_delta);
        }
    }

    // squared error
    double sq = 0.0;
    for (size_t i=0;i<y.size();++i){
        double e = out[i] - y[i];
        sq += e*e;
    }
    return sq;
}