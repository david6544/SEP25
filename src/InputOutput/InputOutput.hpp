#ifndef INPUT_OUTPUT_H
#define INPUT_OUTPUT_H
#include <vector>
#include "../StateSpace/ArrayStateSpace.hpp"

class InputOutput {
protected:
    static InputOutput* instance;
    InputOutput();
public:
    virtual double send_query_recieve_result(const std::vector<int> &query) = 0;
    virtual void output_state(StateSpace &stateSpace) = 0;
    
    static InputOutput* get_instance();
};

#endif // INPUT_OUTPUT_H