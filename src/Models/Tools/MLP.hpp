#ifndef MLP_H
#define MLP_H
#include <vector>
#include <functional>

// -------------------- Type Aliases --------------------
using Vec = std::vector<double>;
using Mat = std::vector<Vec>;

// -------------------- Activation Functions --------------------
struct Activation {
    std::function<double(double)> f;   // activation
    std::function<double(double)> df;  // derivative wrt input
};

// Some standard activations
extern Activation sigmoid;
extern Activation relu;
extern Activation tanh_act;
extern Activation identity;

// -------------------- Layer --------------------
struct Layer {
    int in, out;
    Mat W;  // weights (out x in)
    Vec b;  // biases (out)

    Activation act;

    // Cached forward values
    Vec z;  // pre-activation
    Vec a;  // post-activation

    Layer(int in_, int out_, Activation act_);
    Vec forward(const Vec& x);
};

// -------------------- MLP --------------------
class MLP {
public:
    explicit MLP(const std::vector<int>& sizes,
                 const std::vector<Activation>& acts);

    Vec predict(const Vec& x);
    double train_sample(const Vec& x, const Vec& y, double lr);

private:
    std::vector<Layer> layers;
};
#endif //MPL_H