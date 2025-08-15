#ifndef INPUT_OUTPUT_H
#define INPUT_OUTPUT_H
#include <vector>
#include "../StateSpace/StateSpace.cpp"

class InputOutput {
protected:
    static InputOutput* instance;
    InputOutput() { instance = this; }
public:
    virtual ~InputOutput() = default;
    virtual double send_query_recieve_result(std::vector<int> &query) = 0;
    virtual void output_state(StateSpace &stateSpace) = 0;
    
    static InputOutput* get_instance(){
        return instance;
    }
};

#endif // INPUT_OUTPUT_H