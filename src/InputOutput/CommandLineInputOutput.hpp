#ifndef COMMAND_LINE_IO_H
#define COMMAND_LINE_IO_H
#include <vector>

#include "InputOutput.hpp"

class CommandLineInputOutput : public InputOutput {
    CommandLineInputOutput() = default;
public:
    double send_query_recieve_result(const std::vector<int> &query) override;
    void output_state(const StateSpace &stateSpace) override;
    static void set_IO();
};

#endif // COMMAND_LINE_IO_H