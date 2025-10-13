#ifndef INPUT_OUTPUT_H
#define INPUT_OUTPUT_H
#include <vector>
#include "../Models/Model.hpp"

class InputOutput {
protected:
    static InputOutput* instance;
    InputOutput();
public:
    virtual double send_query_recieve_result(const std::vector<int> &query) = 0;
    virtual void output_state(Model &model) = 0;

    static std::vector<int> index_to_coords(int index, int dimensions, int dimensionSize);
    static InputOutput* get_instance();
};

#endif // INPUT_OUTPUT_H