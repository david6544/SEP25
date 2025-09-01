#ifndef MLP_H
#define MLP_H
#include <vector>
#include <functional>

// -------------------- Type Aliases --------------------
using Mat = std::vector<std::vector<double>>;

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
    std::vector<double> b;  // biases (out)

    Activation act;

    // Cached forward values
    std::vector<double> z;  // pre-activation
    std::vector<double> a;  // post-activation

    Layer(int in_, int out_, Activation act_);
    std::vector<double> forward(const std::vector<double>& x);
};

// -------------------- MLPNetwork --------------------
class MLPNetwork {
public:
    explicit MLPNetwork(const std::vector<int>& sizes,
                 const std::vector<Activation>& acts);

    std::vector<double> predict(const std::vector<double>& x);
    double train_sample(const std::vector<double>& x, const std::vector<double>& y, double lr);

private:
    std::vector<Layer> layers;
};
#endif //MPL_H