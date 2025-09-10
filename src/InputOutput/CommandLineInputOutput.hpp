/**
 * @file CommandLineInputOutput.hpp
 * @author your name (you@domain.com) PLS UPDATE
 * @brief This file declares a CommandLineInputOutput class that handles input and output via the command line.
 * @version 0.1
 * @date 2025-08-27
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#ifndef COMMAND_LINE_IO_H
#define COMMAND_LINE_IO_H
#include <vector>

#include "InputOutput.hpp"

class CommandLineInputOutput : public InputOutput {
    CommandLineInputOutput() = default;
public:
    double send_query_recieve_result(const std::vector<int> &query) override;
    void output_state(Model &model) override;
    static void set_IO();
};

#endif // COMMAND_LINE_IO_H