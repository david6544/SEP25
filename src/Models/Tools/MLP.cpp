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

Activation identity = {
    [](double x){ return x; },
    [](double x){ return 1.0; }
};

Activation relu = {
    [](double x){ return x > 0.0 ? x : 0.0; },
    [](double x){ return x > 0.0 ? 1.0 : 0.0; }
};

Activation tanh_act = {
    [](double x){ return std::tanh(x); },
    [](double x){ double t = std::tanh(x); return 1 - t*t; }
};

Activation ident = {
    [](double x){ return x; },
    [](double){ return 1.0; }
};

// -------------------- Layer --------------------
Layer::Layer(int in_, int out_, Activation act_)
    : in(in_), out(out_), act(act_),
      W(Mat(out_, std::vector<double>(in_))), b(std::vector<double>(out_)),
      z(std::vector<double>(out_)), a(std::vector<double>(out_))
{
    double limit = std::sqrt(6.0 / (in + out));
    for (int i=0;i<out;i++){
        for (int j=0;j<in;j++){
            W[i][j] = rand_uniform(-limit, limit);
        }
        b[i] = 0.0;
    }
}

std::vector<double> Layer::forward(const std::vector<double>& x) {
    assert((int)x.size() == in);
    for (int i=0;i<out;i++){
        double sum = b[i];
        for (int j=0;j<in;j++) sum += W[i][j] * x[j];
        z[i] = sum;
        a[i] = act.f(sum);
    }
    return a;
}

// -------------------- MLPNetwork --------------------
MLPNetwork::MLPNetwork(const std::vector<int>& sizes,
         const std::vector<Activation>& acts)
{
    assert(sizes.size() >= 2);
    assert(acts.size() == sizes.size() - 1);
    for (size_t i=0;i+1<sizes.size();++i){
        layers.emplace_back(sizes[i], sizes[i+1], acts[i]);
    }
}

std::vector<double> MLPNetwork::predict(const std::vector<double>& x) {
    std::vector<double> cur = x;
    for (auto& L : layers) cur = L.forward(cur);
    return cur;
}

double MLPNetwork::train_sample(const std::vector<double>& x, const std::vector<double>& y, double lr) {
    // forward
    std::vector<double> cur = x;
    for (auto& L : layers) cur = L.forward(cur);
    std::vector<double> out = cur;

    assert(out.size() == y.size());

    // output delta
    int Lidx = (int)layers.size() - 1;
    std::vector<double> delta(layers[Lidx].out);
    for (int i=0;i<layers[Lidx].out;i++){
        double err = out[i] - y[i];
        delta[i] = 2.0 * err * layers[Lidx].act.df(layers[Lidx].z[i]);
    }

    // backprop
    for (int li=Lidx; li>=0; --li) {
        Layer& L = layers[li];
        std::vector<double> prev_a = (li==0) ? x : layers[li-1].a;

        // update weights/biases
        for (int i=0;i<L.out;i++){
            for (int j=0;j<L.in;j++){
                double grad_w = delta[i] * prev_a[j];
                L.W[i][j] -= lr * grad_w;
            }
            L.b[i] -= lr * delta[i];
        }

        if (li > 0) {
            std::vector<double> prev_delta(layers[li-1].out, 0.0);
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