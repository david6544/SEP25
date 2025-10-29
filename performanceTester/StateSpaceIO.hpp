#ifndef SS_INPUT_OUTPUT
#define SS_INPUT_OUTPUT

#include "../src/InputOutput/InputOutput.hpp"
#include "FunctionSpace.hpp"

/**
 * IO implementation for model performance testing, inherits from InputOutput
 */
class StateSpaceIO : public InputOutput {
private:
    static FunctionSpace *stateSpace;
    static std::string name;
    static int queries;
    static bool stochastic;
    StateSpaceIO() = default;

public:
    static void set_state_space(FunctionSpace& stateSpace, const std::string& name, int quereies);
    static void set_IO(FunctionSpace& stateSpace, const std::string& name, int quereies, bool stochastic);
    double send_query_recieve_result(const std::vector<int> &query) override;
    void output_state(Model& model, bool outputStateSpace);
    void output_state(Model& model) override;
};

#endif // SS_INPUT_OUTPUT