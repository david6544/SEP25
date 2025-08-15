#ifndef COMMAND_LINE_IO_H
#define COMMAND_LINE_IO_H
#include <vector>

#include "InputOutput.hpp"

class CommandLineInputOutput : public InputOutput {
private:
    CommandLineInputOutput() = default;
public:
    double send_query_recieve_result(std::vector<int> &query) override;
    void output_state(StateSpace &stateSpace) override;
};

#endif // COMMAND_LINE_IO_H