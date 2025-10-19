System Architecture (UML)

This diagram shows the core domain types and how they interact. It’s kept in sync with the current codebase. Update it whenever implementation, class names, or responsibilities change.

```mermaid
%% Rendered by GitHub Mermaid
%% Layout hint
%%{init: {'flowchart': {'diagramPadding': 8} {'defaultRenderer': 'elk'}} }%%

classDiagram
direction TB

%% ===== Core Types =====
class Model {
  <<abstract>>
  - stateSpace : StateSpace*
  + Model(dimensions:int, dimensionSize:int)
  + get_next_query() : std::vector<int>
  + update_prediction(query: std::vector<int>, result: double) : void
  + get_state_space() : StateSpace
}

class StateSpace {
  - stateSpace: vector
  - dimensions: int
  - dimensionsSizes: int[]
  - coords_to_index(coords: std::vector<int>) : int
  + StateSpace(dimensions:int, dimensionSize:int, initialValues:double=0.0)
  + get_dimensions() : int
  + get_dimension_size() : int
  + get(coords: std::vector<int>) : double
  + set(coords: std::vector<int>, value: double) : double
  + get_raw_representation() : std::vector<double>
}

%% ===== Models =====
class DumbModel {
  + DumbModel(dimensions:int, dimensionSize:int)
  + get_next_query() : std::vector<int>
  + update_prediction(query: std::vector<int>, result: double) : void
}
    
class LinearModel {
  + LinearModel(dimensions:int, dimensionSize:int, queries:int)
  + get_next_query() : std::vector<int>
  + update_prediction(query: std::vector<int>, result: double) : void
  + update_prediction_final() : void
  + find_next_nonzero_ix(ix: double) : int
  + find_prev_nonzero_ix(ix: double) : int
  + find_next_nonzero_afterzero_ix(ix: double) : int
  + find_prev_nonzero_afterzero_ix(ix: double) : int
}

class RBFModel {
  <<to be implemented>>
}

class PlaceholderModelN {
  <<to be implemented/extended>>
}

class EnsembleModel {
  <<to be implemented>>
}

%% ===== IO Abstraction =====
class InputOutput {
  <<abstract>>
  - static instance : InputOutput*
  + InputOutput()
  + send_query_recieve_result(query: std::vector<int>) : double
  + output_state(state: StateSpace) : void
  + get_instance() : InputOutput*
}

class CommandLineInputOutput {
  + send_query_recieve_result(query: std::vector<int>) : double
  + output_state(state: StateSpace) : void
  + set_IO() : void
}

%% ===== debugger =====
class DebuggerLogger {
  <<optional>>
  + info(msg: string) : void
  + warn(msg: string) : void
  + metric(key: string, val: double) : void
}

%% ===== Relationships =====
Model o-- StateSpace
Model --> DebuggerLogger : logs metrics/steps
DumbModel ..|> Model
LinearModel ..|> Model
RBFModel ..|> Model
PlaceholderModelN ..|> Model
EnsembleModel ..|> Model
DumbModel ..|> EnsembleModel
LinearModel ..|> EnsembleModel
RBFModel ..|> EnsembleModel
PlaceholderModelN ..|> EnsembleModel
CommandLineInputOutput ..|> InputOutput
InputOutput ..> StateSpace : uses


%% ===== Notes =====
note for StateSpace "Stored as a 1D array (flattened) via coords_to_index(). Avoids N-D container overhead."
note for InputOutput "typo -'recieve' cute little artifact"

```