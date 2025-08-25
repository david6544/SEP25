#ifndef SS_INPUT_OUTPUT
#define SS_INPUT_OUTPUT

#include "../src/InputOutput/InputOutput.hpp"
#include "FunctionSpace.hpp"

class StateSpaceIO : public InputOutput {
private:
    static FunctionSpace *stateSpace;
    StateSpaceIO() = default;
public:
    static void set_state_space(FunctionSpace& stateSpace);
    static void set_IO(FunctionSpace& stateSpace);
    double send_query_recieve_result(const std::vector<int> &query) override;
    void output_state(StateSpace& stateSpace) override;
};

#endif // SS_INPUT_OUTPUT