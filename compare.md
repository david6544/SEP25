diff --git a/CMakeLists.txt b/CMakeLists.txt
index 30e3849..82eda62 100644
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -1,8 +1,10 @@
 cmake_minimum_required(VERSION 3.16)
 project(SEP25 VERSION 0.1.0 LANGUAGES CXX)
 
-cmake_minimum_required(VERSION 3.16)
-project(SEP25 VERSION 0.1.0 LANGUAGES CXX)
+# Selected Algorithm (change text in " ")
+# set(ALGORITHM "DUMB")
+set(ALGORITHM "LINEAR")
+# set(ALGORITHM "RBF")
 
 # Detect if building natively or with a toolchain
 if(CMAKE_TOOLCHAIN_FILE)
@@ -70,10 +72,21 @@ list(FILTER SRC_FILES EXCLUDE REGEX ".*/main.cpp$")
 
 # Main executable with commit hash
 add_executable(sep25_main src/main.cpp ${SRC_FILES})
-set_target_properties(sep25_main PROPERTIES OUTPUT_NAME "sep25_main_${GIT_COMMIT_HASH_SHORT}")
 
-# Include dirs
+message(STATUS "Selected algorithm: ${ALGORITHM}")
+
+# Add compile definition based on chosen algorithm
+target_compile_definitions(sep25_main PUBLIC ${ALGORITHM})
+
 target_include_directories(sep25_main PUBLIC ${CMAKE_BINARY_DIR})
+include_directories(SYSTEM ${CMAKE_CXX_STANDARD_LIBRARIES})
+set_target_properties(sep25_main PROPERTIES OUTPUT_NAME "sep25_main_${GIT_COMMIT_HASH_SHORT}")
+
+# Add performance test executable
+file(GLOB PERFORMANCE_FILES performanceTester/*.cpp)
+add_executable(sep25_performance ${PERFORMANCE_FILES} ${SRC_FILES})
+target_include_directories(sep25_performance PUBLIC ${CMAKE_BINARY_DIR})
+target_compile_definitions(sep25_performance PUBLIC ${ALGORITHM})
 
 # GoogleTest setup (unchanged)
 include(FetchContent)
@@ -88,6 +101,15 @@ FetchContent_MakeAvailable(googletest)
 enable_testing()
 file(GLOB TEST_FILES test/*.cpp)
 add_executable(sep25_tests ${SRC_FILES} ${TEST_FILES})
+target_compile_definitions(sep25_tests PUBLIC "TESTING")
+
+# Link GoogleTest
+target_link_libraries(sep25_tests 
+    gtest_main
+    gmock_main
+)
+
+# Add tests
 target_link_libraries(sep25_tests gtest_main gmock_main)
 add_test(NAME UnitTests COMMAND sep25_tests)
 
diff --git a/README.md b/README.md
index dd20a2c..0328ce6 100644
--- a/README.md
+++ b/README.md
@@ -15,17 +15,27 @@ This project uses CMake as its build system. The project structure includes:
 
 ### MinGW Installation
 1. Install MinGW:
+
+Linux/Ubuntu:
 ```bash
 sudo apt update
 sudo apt install mingw-w64
 ```
 
+For Mac using Homebrew
+```bash
+brew install mingw-w64
+```
+
 2. Validate Installation:
 ```bash
 which x86_64-w64-mingw32-gcc 
 which x86_64-w64-mingw32-g++
 ```
 
+### Changing Models
+In order to change the model that the build is using update the `ALGORITHM` value in the `CMakeLists.txt` file to the Model definition and add the algorithm in main. Please see the DumbModel implementation as an example.
+
 
 ### Local Build Instructions
 
diff --git a/docs/design/uml-system-architecture.md b/docs/design/uml-system-architecture.md
new file mode 100644
index 0000000..0ad3f98
--- /dev/null
+++ b/docs/design/uml-system-architecture.md
@@ -0,0 +1,110 @@
+System Architecture (UML)
+
+This diagram shows the core domain types and how they interact. It’s kept in sync with the current codebase. Update it whenever implementation, class names, or responsibilities change.
+
+```mermaid
+%% Rendered by GitHub Mermaid
+%% Layout hint
+%%{init: {'flowchart': {'diagramPadding': 8} {'defaultRenderer': 'elk'}} }%%
+
+classDiagram
+direction TB
+
+%% ===== Core Types =====
+class Model {
+  <<abstract>>
+  - stateSpace : StateSpace*
+  + Model(dimensions:int, dimensionSize:int)
+  + get_next_query() : std::vector<int>
+  + update_prediction(query: std::vector<int>, result: double) : void
+  + get_state_space() : StateSpace
+}
+
+class StateSpace {
+  - stateSpace: vector
+  - dimensions: int
+  - dimensionsSizes: int[]
+  - coords_to_index(coords: std::vector<int>) : int
+  + StateSpace(dimensions:int, dimensionSize:int, initialValues:double=0.0)
+  + get_dimensions() : int
+  + get_dimension_size() : int
+  + get(coords: std::vector<int>) : double
+  + set(coords: std::vector<int>, value: double) : double
+  + get_raw_representation() : std::vector<double>
+}
+
+%% ===== Models =====
+class DumbModel {
+  + DumbModel(dimensions:int, dimensionSize:int)
+  + get_next_query() : std::vector<int>
+  + update_prediction(query: std::vector<int>, result: double) : void
+}
+    
+class LinearModel {
+  + LinearModel(dimensions:int, dimensionSize:int, queries:int)
+  + get_next_query() : std::vector<int>
+  + update_prediction(query: std::vector<int>, result: double) : void
+  + update_prediction_final() : void
+  + find_next_nonzero_ix(ix: double) : int
+  + find_prev_nonzero_ix(ix: double) : int
+  + find_next_nonzero_afterzero_ix(ix: double) : int
+  + find_prev_nonzero_afterzero_ix(ix: double) : int
+}
+
+class RBFModel {
+  <<to be implemented>>
+}
+
+class PlaceholderModelN {
+  <<to be implemented/extended>>
+}
+
+class EnsembleModel {
+  <<to be implemented>>
+}
+
+%% ===== IO Abstraction =====
+class InputOutput {
+  <<abstract>>
+  - static instance : InputOutput*
+  + InputOutput()
+  + send_query_recieve_result(query: std::vector<int>) : double
+  + output_state(state: StateSpace) : void
+  + get_instance() : InputOutput*
+}
+
+class CommandLineInputOutput {
+  + send_query_recieve_result(query: std::vector<int>) : double
+  + output_state(state: StateSpace) : void
+  + set_IO() : void
+}
+
+%% ===== debugger =====
+class DebuggerLogger {
+  <<optional>>
+  + info(msg: string) : void
+  + warn(msg: string) : void
+  + metric(key: string, val: double) : void
+}
+
+%% ===== Relationships =====
+Model o-- StateSpace
+Model --> DebuggerLogger : logs metrics/steps
+DumbModel ..|> Model
+LinearModel ..|> Model
+RBFModel ..|> Model
+PlaceholderModelN ..|> Model
+EnsembleModel ..|> Model
+DumbModel ..|> EnsembleModel
+LinearModel ..|> EnsembleModel
+RBFModel ..|> EnsembleModel
+PlaceholderModelN ..|> EnsembleModel
+CommandLineInputOutput ..|> InputOutput
+InputOutput ..> StateSpace : uses
+
+
+%% ===== Notes =====
+note for StateSpace "Stored as a 1D array (flattened) via coords_to_index(). Avoids N-D container overhead."
+note for InputOutput "typo -'recieve' cute little artifact"
+
+```
\ No newline at end of file
diff --git a/michelle_test_lines.md b/michelle_test_lines.md
index bd3abc5..dc44a42 100644
--- a/michelle_test_lines.md
+++ b/michelle_test_lines.md
@@ -7,7 +7,7 @@ seq 0.1 0.1 0.7 | ./build/sep25_main_dev 1 10 7
 ```
 
 ```bash
-seq 0.1 0.1 0.7 | wine ./build-windows/sep25_main_659ccf5.exe 1 10 7
+seq 0.1 0.1 0.7 | ./build-windows/sep25_main_fa2f6d3.exe 1 10 7
 ```
 
 ### 2. **1-D, 100 elements, 70 queries**
@@ -40,71 +40,4 @@ mkdir -p build && cd build && cmake .. && cmake --build . && cd ..
 
 ```bash
 rm -rf build-windows && cmake -B build-windows -S . -DCMAKE_TOOLCHAIN_FILE=windows-toolchain.cmake -DCMAKE_BUILD_TYPE=Release &&  cmake --build build-windows
-```
-
-# ---------
-In CMakeList.txt we have this part:
-
-### 1️⃣ How the commit hash is generated in your CMake
-
-From your `CMakeLists.txt`:
-
-```cmake
-execute_process(
-    COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
-    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
-    OUTPUT_VARIABLE GIT_COMMIT_HASH_SHORT
-    OUTPUT_STRIP_TRAILING_WHITESPACE
-)
-```
-
-* `git rev-parse --short HEAD` returns the **short SHA of the current commit** in your Git repository.
-
-  * Example: `659ccf5` is a 7-character prefix of a full commit hash like `659ccf5a1b2c3d4e5f6g7h8i9j0klmnop`.
-* This short hash is then appended to the executable name:
-
-```cmake
-set_target_properties(sep25_main PROPERTIES OUTPUT_NAME "sep25_main_${GIT_COMMIT_HASH_SHORT}")
-```
-
-So `sep25_main_659ccf5.exe` literally means:
-
-```
-sep25_main_<short Git commit hash>.exe
-```
-
----
-
-### 2️⃣ What this hash tells you
-
-* It **identifies the exact commit of your source code** used to build the executable.
-* It does **not** directly encode:
-
-  * The model type (LINEAR, RBF, DUMB)
-  * The parameters or state of the model
-  * Any runtime configuration
-
----
-
-### 3️⃣ How to know which model was used
-
-In your CMake:
-
-```cmake
-# Selected Algorithm
-set(ALGORITHM "LINEAR")
-# set(ALGORITHM "RBF")
-# set(ALGORITHM "DUMB")
-
-# Add compile definition based on chosen algorithm
-target_compile_definitions(sep25_main PUBLIC ${ALGORITHM})
-```
-
-* The chosen algorithm is compiled **into the binary at build time** via a preprocessor definition (`LINEAR`, `RBF`, or `DUMB`).
-* So the **binary name itself doesn’t tell you the algorithm**. You’d need one of these:
-
-  1. **Check the commit in Git** (`git show 659ccf5`) to see which `ALGORITHM` was active.
-  2. **Print it at runtime** (your code could log `ALGORITHM`).
-  3. **Embed it in the executable name manually** during build.
-
----
+```
\ No newline at end of file
diff --git a/performanceTester/FunctionList.hpp b/performanceTester/FunctionList.hpp
new file mode 100644
index 0000000..b9830af
--- /dev/null
+++ b/performanceTester/FunctionList.hpp
@@ -0,0 +1,57 @@
+#pragma once
+#include<vector>
+#include<math.h>
+
+namespace testfunctions {
+
+double ackleyFunction(const std::vector<int>& query){
+    const double a = 20.0;
+    const double b = 0.2;
+    const double c = 2 * M_PI;
+
+    size_t n = query.size();
+    if (n == 0) return 0.0;
+
+    double sumSquares = 0.0;
+    double sumCos = 0.0;
+
+    for (int x : query) {
+        double xd = static_cast<double>(x);  // convert int to double
+        sumSquares += xd * xd;
+        sumCos += cos(c * xd);
+    }
+
+    double term1 = -a * exp(-b * sqrt(sumSquares / n));
+    double term2 = -exp(sumCos / n);
+
+    return term1 + term2 + a + exp(1);
+}
+
+double sumpow (const std::vector<int>& query) {
+    double sumSquares = 0.0;
+    for (int i = 1; i <= query.size(); i++) {
+        sumSquares += pow(abs(query[i - 1]),i);
+    }
+    return sumSquares;
+}
+
+double griewank(const std::vector<int>& query) {
+    double sumTerm1 = 0.0;
+    double sumTerm2 = 1.0;
+    for (int i = 1; i <= query.size(); i++) {
+        int x = query[i - 1];
+        sumTerm1 += pow(x,2) / 4000;
+        sumTerm2 *= cos(x/sqrt(i));
+    }
+    return sumTerm1 - sumTerm2 + 1;
+}
+
+double rastrigin(const std::vector<int>& query) {
+    double dimTerm = query.size() * 10;
+    double sumTerm = 0.0;
+    for (int x : query) {
+        sumTerm += pow(x,2) - 10 * cos(2* M_PI * x );
+    }
+    return dimTerm + sumTerm;
+}
+}
\ No newline at end of file
diff --git a/performanceTester/FunctionSpace.cpp b/performanceTester/FunctionSpace.cpp
new file mode 100644
index 0000000..e131ae8
--- /dev/null
+++ b/performanceTester/FunctionSpace.cpp
@@ -0,0 +1,73 @@
+#include "FunctionSpace.hpp"
+
+#include <exception>
+#include <queue>
+
+FunctionSpace::FunctionSpace(int dimensions, int dimensionSize, SpaceFunctionType spaceFunction) : 
+    StateSpace(dimensions, dimensionSize), 
+    spaceFunction(spaceFunction) {}
+
+int FunctionSpace::get_dimensions() const {
+    return this->dimensions;
+}
+
+int FunctionSpace::get_dimension_size() const {
+    return this->dimensionSize;
+}
+
+double FunctionSpace::get(const std::vector<int>& coords) const {
+    for (auto i : coords)
+        std::cout << i << " ";
+    std::cout << std::endl;
+    if (coords.size() != dimensions)
+        throw std::out_of_range("query dimension size is out of range");
+    
+    for (auto val : coords)
+        if (val < 0 || val >= dimensionSize)
+            throw std::out_of_range("query values are out of range");
+    
+    return this->spaceFunction(coords);
+}
+
+
+
+void FunctionSpace::getResultsHelper(Model& model, std::vector<int>& query, int dimension){
+    for (auto i : query)
+        std::cout << i << " ";
+    std::cout << std::endl;
+    
+    if (dimension == query.size()){
+        return;
+    }
+
+    for (int i = 1; i < dimensionSize; i++){
+        query[dimension] = i;
+
+        double actualResult = this->get(query);
+        double predictedResult = model.get_value_at(query);
+
+        this->results.updateResults(actualResult, predictedResult);
+
+        getResultsHelper(model, query, dimension+1);
+    }
+    query[dimension] = 0;
+    
+    return;
+}
+
+Results FunctionSpace::getResults(Model& model) {
+    results = Results();
+
+    std::vector<int> initialQuery(dimensions, 0);
+
+    double actualResult = this->get(initialQuery);
+    double predictedResult = model.get_value_at(initialQuery);
+
+    this->results.updateResults(actualResult, predictedResult);
+
+    getResultsHelper(model, initialQuery, 0);
+
+    this->allResults.push_back(this->results);
+    
+    return this->results;
+}
\ No newline at end of file
diff --git a/performanceTester/FunctionSpace.hpp b/performanceTester/FunctionSpace.hpp
new file mode 100644
index 0000000..1461813
--- /dev/null
+++ b/performanceTester/FunctionSpace.hpp
@@ -0,0 +1,84 @@
+#ifndef FUNCTION_SPACE
+#define FUNCTION_SPACE
+
+#include "../src/StateSpace/ArrayStateSpace.hpp"
+#include "../src/Models/Model.hpp"
+#include <vector>
+#include <functional>
+#include <algorithm>
+#include <limits>
+
+#define SpaceFunctionType std::function<double(const std::vector<int>&)>
+
+typedef struct Results {
+    int correct = 0;
+    int totalSeen = 0;
+
+    double sumAbsoluteError = 0;
+    double sumSquaredError = 0;
+
+    double predictedResultSum = 0;
+
+    double minActual = std::numeric_limits<double>::max();
+    double maxActual = std::numeric_limits<double>::lowest();
+    double minPredicted = std::numeric_limits<double>::max();
+    double maxPredicted = std::numeric_limits<double>::lowest();
+
+    void updateResults(double actual, double predicted){
+        using namespace std;
+        totalSeen++;
+        if (predicted >= actual * 0.99 && predicted <= actual * 1.01){
+            this->correct++;
+        }
+        double error = abs(actual-predicted);
+        sumAbsoluteError += error;
+        sumSquaredError += pow(error, 2);
+
+        minActual = min(minActual, actual);
+        maxActual = max(maxActual, actual);
+
+        minPredicted = min(minPredicted, predicted);
+        maxPredicted = max(maxPredicted, predicted);
+
+        predictedResultSum += predicted;
+    }
+    double percentCorrect() const {
+        return totalSeen > 0 ? static_cast<double>(correct) / totalSeen * 100.0 : 0.0;
+    }
+
+    double meanAbsoluteError() const {
+        return totalSeen > 0 ? sumAbsoluteError / totalSeen : 0.0;
+    }
+
+    double rootMeanSquaredError() const {
+        return totalSeen > 0 ? std::sqrt(sumSquaredError / totalSeen) : 0.0;
+    }
+
+    double meanPredicted() const {
+        return totalSeen > 0 ? predictedResultSum / totalSeen : 0.0;
+    }
+} Results;
+
+class FunctionSpace : public StateSpace {
+private:
+    SpaceFunctionType spaceFunction;
+    Results results;
+
+    std::vector<Results> allResults;
+    
+    void getResultsHelper(Model& model, std::vector<int>& query, int dimension);
+public:
+    FunctionSpace(int dimensions, int dimensionSize, SpaceFunctionType spaceFunction);
+
+    int get_dimensions() const override;
+
+    int get_dimension_size() const override;
+    
+    double get(const std::vector<int>& coords) const override;
+
+    Results getResults(Model& model);
+    std::vector<Results> getAllResults(){ return allResults; };
+};
+
+
+#endif //FUNCTION_SPACE
\ No newline at end of file
diff --git a/performanceTester/StateSpaceIO.cpp b/performanceTester/StateSpaceIO.cpp
new file mode 100644
index 0000000..0781fb8
--- /dev/null
+++ b/performanceTester/StateSpaceIO.cpp
@@ -0,0 +1,33 @@
+#include "StateSpaceIO.hpp"
+#include <iostream>
+using std::cout, std::endl;
+
+FunctionSpace* StateSpaceIO::stateSpace = nullptr;
+
+void StateSpaceIO::set_IO(FunctionSpace& stateSpace){
+    if (instance == nullptr)
+        StateSpaceIO::stateSpace = &stateSpace;
+        instance = new StateSpaceIO;
+}
+
+void StateSpaceIO::set_state_space(FunctionSpace& stateSpace){
+    StateSpaceIO::stateSpace = &stateSpace;
+}
+
+double StateSpaceIO::send_query_recieve_result(const std::vector<int> &query) {
+    double result = stateSpace->get(query);
+    return result;
+}
+
+void StateSpaceIO::output_state(Model &model) {
+    Results results = stateSpace->getResults(model);
+
+    cout << "Performance Metrics:" << endl;
+    cout << "-------------------" << endl;
+    cout << "Percent Correct: " << results.percentCorrect() << "%" << endl;
+    cout << "Mean Absolute Error: " << results.meanAbsoluteError() << endl;
+    cout << "Root Mean Squared Error: " << results.rootMeanSquaredError() << endl;
+    cout << "Mean Predicted Value: " << results.meanPredicted() << endl;
+    cout << "Actual Min/Max: [" << results.minActual << ", " << results.maxActual << "]" << endl;
+    cout << "Predicted Min/Max: [" << results.minPredicted << ", " << results.maxPredicted << "]" << endl;
+}
\ No newline at end of file
diff --git a/performanceTester/StateSpaceIO.hpp b/performanceTester/StateSpaceIO.hpp
new file mode 100644
index 0000000..8261503
--- /dev/null
+++ b/performanceTester/StateSpaceIO.hpp
@@ -0,0 +1,19 @@
+#ifndef SS_INPUT_OUTPUT
+#define SS_INPUT_OUTPUT
+
+#include "../src/InputOutput/InputOutput.hpp"
+#include "FunctionSpace.hpp"
+
+class StateSpaceIO : public InputOutput {
+private:
+    static FunctionSpace *stateSpace;
+    StateSpaceIO() = default;
+
+public:
+    static void set_state_space(FunctionSpace& stateSpace);
+    static void set_IO(FunctionSpace& stateSpace);
+    double send_query_recieve_result(const std::vector<int> &query) override;
+    void output_state(Model& model) override;
+};
+
+#endif // SS_INPUT_OUTPUT
\ No newline at end of file
diff --git a/performanceTester/main.cpp b/performanceTester/main.cpp
new file mode 100644
index 0000000..c20b69e
--- /dev/null
+++ b/performanceTester/main.cpp
@@ -0,0 +1,93 @@
+
+#if defined(LINEAR)
+    #include "../src/Models/LinearModel.hpp"
+    #define CurrentModel LinearModel
+#elif defined(DUMB)
+    #include "../src/Models/DumbModel.hpp"
+    #define CurrentModel DumbModel
+#elif defined(RBF)
+    #include "../src/Models/RBFModel.hpp"
+    #define CurrentModel RBFModel
+#else
+    #error "Algorthim was not defined please check readme for build instructions"
+#endif
+
+#include "FunctionList.hpp"
+#include "FunctionSpace.hpp"
+#include "StateSpaceIO.hpp"
+
+#include <iostream>
+
+
+//This is gpt'd lmao,
+void output_performance(std::vector<Results> results) {
+    std::vector<double> errors;
+    std::cout << "\n \n";
+
+    for (const auto& res : results) {
+        // Use mean absolute error as a representative error for each Results
+        errors.push_back(res.meanAbsoluteError());
+    }
+    if (errors.empty()) {
+        std::cout << "No error data available." << std::endl;
+        return;
+    }
+    std::sort(errors.begin(), errors.end());
+    double median;
+    size_t n = errors.size();
+    if (n % 2 == 0) {
+        median = (errors[n/2 - 1] + errors[n/2]) / 2.0;
+    } else {
+        median = errors[n/2];
+    }
+    std::cout << "Median of mean absolute errors: " << median << std::endl;
+}
+
+void runSeveral(int dimensions, int dimensionSize, SpaceFunctionType func) {
+    std::vector<double> querySizes = {0.10, 0.25, 0.50, 0.75};
+    std::vector<Results> results;
+    for (auto size: querySizes) {
+        int queryCount = dimensionSize / size;
+        
+        FunctionSpace fspace(dimensions, dimensionSize, func);
+        StateSpaceIO::set_IO(fspace);
+        InputOutput* io = InputOutput::get_instance();
+        
+        CurrentModel model(dimensions, dimensionSize, queryCount);
+        
+        for (int i = 0; i < queryCount; i++){
+            std::vector<int> query = model.get_next_query();
+            double result = io->send_query_recieve_result(query);
+            model.update_prediction(query, result);
+        }
+        results.emplace_back(fspace.getResults(model));
+        io->output_state(model);
+    }
+    output_performance(results);
+
+}
+
+void runSingle(int dimensions, int dimensionSize, int queries, SpaceFunctionType func) {
+    
+    FunctionSpace fspace(dimensions, dimensionSize, func);
+    StateSpaceIO::set_IO(fspace);
+    InputOutput* io = InputOutput::get_instance();
+    
+    CurrentModel model(dimensions, dimensionSize, queries);
+    
+    for (int i = 0; i < queries; i++){
+        std::vector<int> query = model.get_next_query();
+        double result = io->send_query_recieve_result(query);
+        model.update_prediction(query, result);
+    }
+    io->output_state(model);
+}
+
+
+int main(void) {
+
+    int dimensions = 1, dimensionSize = 300, queries = 50;
+    auto func = testfunctions::griewank;
+    //runSeveral(dimensions,dimensionSize, func);
+    runSingle(dimensions, dimensionSize, queries, func);
+}
diff --git a/src/InputOutput/CommandLineInputOutput.cpp b/src/InputOutput/CommandLineInputOutput.cpp
index 064f234..2ff8842 100644
--- a/src/InputOutput/CommandLineInputOutput.cpp
+++ b/src/InputOutput/CommandLineInputOutput.cpp
@@ -1,6 +1,7 @@
 #include <iostream>
 
 #include "CommandLineInputOutput.hpp"
+#include <cmath>
 
 void CommandLineInputOutput::set_IO(){
     if (instance == nullptr)
@@ -19,10 +20,31 @@ double CommandLineInputOutput::send_query_recieve_result(const std::vector<int>
     return result;
 }
 
-void CommandLineInputOutput::output_state(const StateSpace &stateSpace){
-    std::vector<double> vec2Output = stateSpace.get_raw_representation();
-    if (vec2Output.size() == 0) return;
-    for (int i = 0; i < vec2Output.size(); i++)
-        std::cout << vec2Output[i] << ((i == vec2Output.size()-1) ? "\n" : " ");
+std::vector<int> index_to_coords(int index, int dimensions, int dimensionSize) {
+    std::vector<int> coords(dimensions);
+    for (int d = dimensions - 1; d >= 0; --d) {
+        coords[d] = index % dimensionSize;
+        index /= dimensionSize;
+    }
+    if (index != 0) {
+        throw std::out_of_range("index too large for given dimensions");
+    }
+    return coords;
+}
 
+void CommandLineInputOutput::output_state(Model &model){
+    int dimensions = model.get_dimensions();
+    int dimensionSize = model.get_dimensionSize();
+
+    long long maxIdx = 1;
+    for (int i = 0; i < dimensions; ++i)
+        maxIdx *= dimensionSize;
+
+    auto coords = index_to_coords(0, dimensions, dimensionSize);
+    std::cout << model.get_value_at(coords);
+    for (long long i = 1; i < maxIdx; i++) {
+        coords = index_to_coords(i, dimensions, dimensionSize);
+        std::cout << " " << model.get_value_at(coords);
+    }
+    std::cout << std::endl;
 }
\ No newline at end of file
diff --git a/src/InputOutput/CommandLineInputOutput.hpp b/src/InputOutput/CommandLineInputOutput.hpp
index b6874a1..9cf219e 100644
--- a/src/InputOutput/CommandLineInputOutput.hpp
+++ b/src/InputOutput/CommandLineInputOutput.hpp
@@ -8,7 +8,7 @@ class CommandLineInputOutput : public InputOutput {
     CommandLineInputOutput() = default;
 public:
     double send_query_recieve_result(const std::vector<int> &query) override;
-    void output_state(const StateSpace &stateSpace) override;
+    void output_state(Model &model) override;
     static void set_IO();
 };
 
diff --git a/src/InputOutput/InputOutput.cpp b/src/InputOutput/InputOutput.cpp
index bc23669..4b73f3f 100644
--- a/src/InputOutput/InputOutput.cpp
+++ b/src/InputOutput/InputOutput.cpp
@@ -1,5 +1,7 @@
 #include "InputOutput.hpp"
 
+#include <stdexcept>
+
 InputOutput* InputOutput::instance = nullptr;
 InputOutput::InputOutput() {
     if (instance == nullptr)
diff --git a/src/InputOutput/InputOutput.hpp b/src/InputOutput/InputOutput.hpp
index 4168f7f..04f9ae3 100644
--- a/src/InputOutput/InputOutput.hpp
+++ b/src/InputOutput/InputOutput.hpp
@@ -1,7 +1,7 @@
 #ifndef INPUT_OUTPUT_H
 #define INPUT_OUTPUT_H
 #include <vector>
-#include "../StateSpace/StateSpace.hpp"
+#include "../Models/Model.hpp"
 
 class InputOutput {
 protected:
@@ -9,7 +9,7 @@ protected:
     InputOutput();
 public:
     virtual double send_query_recieve_result(const std::vector<int> &query) = 0;
-    virtual void output_state(const StateSpace &stateSpace) = 0;
+    virtual void output_state(Model &model) = 0;
     
     static InputOutput* get_instance();
 };
diff --git a/src/Models/DumbModel.cpp b/src/Models/DumbModel.cpp
index b46d3ed..b1d95dc 100644
--- a/src/Models/DumbModel.cpp
+++ b/src/Models/DumbModel.cpp
@@ -1,10 +1,12 @@
+#if defined(DUMB) || defined(TESTING)
 #include <vector>
 #include <random>
 #include <ctime>
 
 #include "DumbModel.hpp"
 
-DumbModel::DumbModel(int dimensions, int dimensionSize) : Model(dimensions, dimensionSize) {
+DumbModel::DumbModel(int dimensions, int dimensionSize, int totalQueries) : 
+    Model(dimensions, dimensionSize, totalQueries), stateSpace(new ArrayStateSpace(dimensions, dimensionSize)) {
     std::srand(std::time(nullptr));
 }
 
@@ -24,3 +26,8 @@ void DumbModel::update_prediction(const std::vector<int> &query, double result)
     stateSpace->set(query, result);
 }
 
+double DumbModel::get_value_at(const std::vector<int> &query) {
+    return this->stateSpace->get(query);
+}
+
+#endif // DUMB || TESTING
\ No newline at end of file
diff --git a/src/Models/DumbModel.hpp b/src/Models/DumbModel.hpp
index 8f690ee..b02c2ae 100644
--- a/src/Models/DumbModel.hpp
+++ b/src/Models/DumbModel.hpp
@@ -1,7 +1,9 @@
+#if defined(DUMB) || defined(TESTING)
 #ifndef DUMB_MODEL_H
 #define DUMB_MODEL_H
 
 #include "Model.hpp"
+#include "../StateSpace/ArrayStateSpace.hpp"
 
 /**
  * @brief The DumbModel is merely for testing the client - It 
@@ -9,11 +11,14 @@
  * 
  */
 class DumbModel : public Model {
+    ArrayStateSpace* stateSpace;
 public:
-    DumbModel(int dimensions, int dimensionSize);
+    DumbModel(int dimensions, int dimensionSize, int totalQueries);
     std::vector<int> get_next_query() override;
     void update_prediction(const std::vector<int> &query, double result) override;
+    double get_value_at(const std::vector<int> &query) override;
 };
 
 
-#endif //DUMB_MODEL_H
\ No newline at end of file
+#endif // DUMB_MODEL_H
+#endif // DUMB || TESTING
\ No newline at end of file
diff --git a/src/Models/LinearModel.cpp b/src/Models/LinearModel.cpp
index 7a1fdac..d88b0c9 100644
--- a/src/Models/LinearModel.cpp
+++ b/src/Models/LinearModel.cpp
@@ -1,22 +1,25 @@
-
+#if defined(LINEAR) || defined(TESTING)
 #include <vector>
 #include <random>
 #include <ctime>
 
 #include "LinearModel.hpp"
 
-LinearModel::LinearModel(int dimensions, int dimensionSize, int queries) : Model(dimensions, dimensionSize), m_maxQueryCount(queries) {
+LinearModel::LinearModel(int dimensions, int dimensionSize, int totalQueries) : 
+    Model(dimensions, dimensionSize, totalQueries), stateSpace(new ArrayStateSpace(dimensions, dimensionSize)) {
     std::srand(std::time(nullptr));
 }
 
 std::vector<int> LinearModel::get_next_query() {
+    currentQuery++;
+
     static int traversed_ix = 0;
 
     int D = this->stateSpace->get_dimensions();
     int K = this->stateSpace->get_dimension_size();
     std::vector<int> nextQuery(D, 0);
 
-    double step = ((double)(K-1))/(double)m_maxQueryCount;
+    double step = ((double)(K-1))/(double)totalQueries;
     for (auto &var : nextQuery){
         double vard = step/2.0 + ((double)traversed_ix) * step;
         var = (int)std::round(vard);
@@ -28,7 +31,7 @@ std::vector<int> LinearModel::get_next_query() {
     return nextQuery;
 }
 
-void print_statespace(StateSpace& stateSpace) {
+void print_statespace(ArrayStateSpace& stateSpace) {
     std::vector<double> vec2Output = stateSpace.get_raw_representation();
     if (vec2Output.size() == 0) return;
     for (int i = 0; i < vec2Output.size(); i++)
@@ -91,6 +94,10 @@ int LinearModel::find_prev_nonzero_afterzero_ix(double ix) {
     return traversed_ix;
 }
 
+double LinearModel::get_value_at(const std::vector<int> &query) {
+    return this->stateSpace->get(query);
+}
+
 void LinearModel::update_prediction_final() {
     
     int D = this->stateSpace->get_dimensions();
@@ -138,5 +145,9 @@ void LinearModel::update_prediction_final() {
 
 void LinearModel::update_prediction(const std::vector<int> &query, double result) {
     stateSpace->set(query, result);
+    if (currentQuery == totalQueries)
+        update_prediction_final();
 }
 
+#endif // LINEAR || TESTING
+
diff --git a/src/Models/LinearModel.hpp b/src/Models/LinearModel.hpp
index 2b69453..49bdc43 100644
--- a/src/Models/LinearModel.hpp
+++ b/src/Models/LinearModel.hpp
@@ -1,23 +1,25 @@
-
+#if defined(LINEAR) || defined(TESTING)
 #ifndef LINEAR_MODEL_H
 #define LINEAR_MODEL_H
 
 #include "Model.hpp"
+#include "../StateSpace/ArrayStateSpace.hpp"
 
 class LinearModel : public Model {
+    ArrayStateSpace* stateSpace;
 public:
-    LinearModel(int dimensions, int dimensionSize, int queries);
+    LinearModel(int dimensions, int dimensionSize, int totalQueries);
     std::vector<int> get_next_query() override;
     void update_prediction(const std::vector<int> &query, double result) override;
+    double get_value_at(const std::vector<int> &query) override;
+private:
     void update_prediction_final();
     int find_next_nonzero_ix(double ix);
     int find_prev_nonzero_ix(double ix);
     int find_next_nonzero_afterzero_ix(double ix);
     int find_prev_nonzero_afterzero_ix(double ix);
-
-private:
-    int m_maxQueryCount = 0;
 };
 
 
-#endif //LINEAR_MODEL_H
\ No newline at end of file
+#endif //LINEAR_MODEL_H
+#endif // LINEAR || TESTING
\ No newline at end of file
diff --git a/src/Models/Model.hpp b/src/Models/Model.hpp
index 8b92adf..1d89f76 100644
--- a/src/Models/Model.hpp
+++ b/src/Models/Model.hpp
@@ -2,16 +2,17 @@
 #define MODEL_H
 #include <vector>
 
-#include "../StateSpace/StateSpace.hpp"
-
 class Model {
 protected:
-    StateSpace* stateSpace = nullptr;
+    int dimensions, dimensionSize, totalQueries, currentQuery;
 public:
-        Model(int dimensions, int dimensionSize): stateSpace(new StateSpace(dimensions, dimensionSize)) {};
+        Model(int dimensions, int dimensionSize, int totalQueries): 
+            totalQueries(totalQueries), dimensions(dimensions), dimensionSize(dimensionSize) {};
+        int get_dimensions(){ return dimensions; }
+        int get_dimensionSize(){ return dimensionSize; }
         virtual std::vector<int> get_next_query() = 0;
         virtual void update_prediction(const std::vector<int> &query, double result) = 0;
-        StateSpace get_state_space() const { return *stateSpace; };
+        virtual double get_value_at(const std::vector<int> &query) = 0;
 };
 
 
diff --git a/src/Models/RBFModel.cpp b/src/Models/RBFModel.cpp
new file mode 100644
index 0000000..d98720c
--- /dev/null
+++ b/src/Models/RBFModel.cpp
@@ -0,0 +1,115 @@
+#if defined(RBF) || defined(TESTING)
+#include "RBFModel.hpp"
+#include <cmath>
+#include <limits>
+#include <functional>
+
+RBFModel::RBFModel(int dimensions, int dimensionSize, int totalQueries)
+    : Model(dimensions, dimensionSize, totalQueries),
+      stateSpace(new ArrayStateSpace(dimensions, dimensionSize)),
+      m_sigma(1.0) {}
+
+std::vector<std::vector<int>> RBFModel::generate_all_points() const {
+    std::vector<std::vector<int>> points;
+    std::vector<int> current(stateSpace->get_dimensions(), 0);
+
+    std::function<void(int)> dfs = [&](int dim) {
+        if (dim == stateSpace->get_dimensions()) {
+            points.push_back(current);
+            return;
+        }
+        for (int i = 0; i < stateSpace->get_dimension_size(); i++) {
+            current[dim] = i;
+            dfs(dim + 1);
+        }
+    };
+
+    dfs(0);
+    return points;
+}
+
+bool RBFModel::is_queried(const std::vector<int>& point) const {
+    for (const auto& [q, _] : m_data) {
+        if (q == point) return true;
+    }
+    return false;
+}
+
+std::vector<int> RBFModel::get_next_query() {
+    currentQuery++;
+
+    auto candidates = generate_all_points();
+
+    // If no queries yet, pick the first point
+    if (m_data.empty()) {
+        return candidates.front();
+    }
+
+    std::vector<int> bestPoint;
+    double bestScore = -1.0;
+
+    for (const auto& point : candidates) {
+        if (is_queried(point)) continue;
+
+        double minDist2 = std::numeric_limits<double>::infinity();
+        for (const auto& [q, _] : m_data) {
+            double dist2 = 0.0;
+            for (size_t i = 0; i < point.size(); i++) {
+                double d = static_cast<double>(point[i]) - static_cast<double>(q[i]);
+                dist2 += d * d;
+            }
+            if (dist2 < minDist2) minDist2 = dist2;
+        }
+
+        if (minDist2 > bestScore) {
+            bestScore = minDist2;
+            bestPoint = point;
+        }
+    }
+
+    return bestPoint;
+}
+
+void RBFModel::update_prediction(const std::vector<int> &query, double result) {
+    m_data.emplace_back(query, result);
+    stateSpace->set(query, result);
+
+    if (currentQuery == totalQueries)
+        update_prediction_final();
+}
+
+double RBFModel::predict(const std::vector<int> &point) const {
+    if (m_data.empty()) return 0.0;
+
+    double num = 0.0;
+    double den = 0.0;
+
+    for (const auto& [q, val] : m_data) {
+        double dist2 = 0.0;
+        for (size_t i = 0; i < point.size(); i++) {
+            double d = static_cast<double>(point[i]) - static_cast<double>(q[i]);
+            dist2 += d * d;
+        }
+        double w = std::exp(-dist2 / (2 * m_sigma * m_sigma));
+        num += w * val;
+        den += w;
+    }
+
+    return (den > 0.0 ? num / den : 0.0);
+}
+
+void RBFModel::update_prediction_final() {
+    auto all_points = generate_all_points();
+    for (const auto& p : all_points) {
+        if (!is_queried(p)) {
+            double val = predict(p);
+            stateSpace->set(p, val);
+        }
+    }
+}
+
+double RBFModel::get_value_at(const std::vector<int> &query) {
+    return this->stateSpace->get(query);
+}
+
+#endif // RBF || TESTING
diff --git a/src/Models/RBFModel.hpp b/src/Models/RBFModel.hpp
new file mode 100644
index 0000000..2bc840f
--- /dev/null
+++ b/src/Models/RBFModel.hpp
@@ -0,0 +1,31 @@
+#if defined(RBF) || defined(TESTING)
+#ifndef RBF_MODEL_H
+#define RBF_MODEL_H
+
+#include "Model.hpp"
+#include "../StateSpace/ArrayStateSpace.hpp"
+#include <vector>
+#include <utility>
+
+class RBFModel : public Model {
+    ArrayStateSpace* stateSpace;
+
+public:
+    RBFModel(int dimensions, int dimensionSize, int totalQueries);
+
+    std::vector<int> get_next_query() override;
+    void update_prediction(const std::vector<int> &query, double result) override;
+    double get_value_at(const std::vector<int> &query) override;
+
+private:
+    double m_sigma;
+    std::vector<std::pair<std::vector<int>, double>> m_data;
+
+    std::vector<std::vector<int>> generate_all_points() const;
+    bool is_queried(const std::vector<int>& point) const;
+    double predict(const std::vector<int> &point) const;
+    void update_prediction_final();
+};
+
+#endif // RBF_MODEL_H
+#endif // RBF || TESTING
diff --git a/src/StateSpace/ArrayStateSpace.cpp b/src/StateSpace/ArrayStateSpace.cpp
new file mode 100644
index 0000000..2de5fba
--- /dev/null
+++ b/src/StateSpace/ArrayStateSpace.cpp
@@ -0,0 +1,43 @@
+#include "ArrayStateSpace.hpp"
+
+#include <string>
+
+int ArrayStateSpace::coords_to_index(const std::vector<int>& coords) const {
+    if (coords.size() != this->dimensions) 
+        throw std::invalid_argument("Index has Invalid Number of Dimensions");
+    int rawIndex = 0;
+    int multiplier = 1;
+    for (int i = dimensions - 1; i > -1; i--) {
+        if (coords[i] < 0 || coords[i] >= this->dimensionSize) 
+            throw std::out_of_range("Coordinate at index [" + std::to_string(i) + "] out of range");
+        rawIndex += coords[i] * multiplier;
+        multiplier *= dimensionSize;
+    }
+    return rawIndex;
+}
+
+ArrayStateSpace::ArrayStateSpace(int dimensions, int dimensionSize, double initialValues) :
+        StateSpace(dimensions, dimensionSize),
+        stateSpaceArray(std::pow(dimensionSize, dimensions), initialValues)
+    {}
+
+
+int ArrayStateSpace::get_dimensions() const {
+    return this->dimensions;
+}
+
+int ArrayStateSpace::get_dimension_size() const {
+    return this->dimensionSize;
+}
+
+double ArrayStateSpace::get(const std::vector<int>& coords) const {
+    return this->stateSpaceArray[coords_to_index(coords)];
+}
+
+std::vector<double> ArrayStateSpace::get_raw_representation() const {
+    return this->stateSpaceArray;
+}
+
+double ArrayStateSpace::set(const std::vector<int>& coords, double value) {
+    return this->stateSpaceArray[coords_to_index(coords)] = value;
+}
\ No newline at end of file
diff --git a/src/StateSpace/ArrayStateSpace.hpp b/src/StateSpace/ArrayStateSpace.hpp
new file mode 100644
index 0000000..df3f834
--- /dev/null
+++ b/src/StateSpace/ArrayStateSpace.hpp
@@ -0,0 +1,38 @@
+/* 
+
+ArrayArrayStateSpace.hpp
+Implements a ArrayStateSpace class that allows for interactions with an n-dimensional integer indexed array.
+
+*/
+
+
+#ifndef ARRAY_STATE_SPACE_H
+#define ARRAY_STATE_SPACE_H
+#include <vector>
+#include <iostream>
+#include <cmath>
+
+#include "StateSpace.hpp"
+
+class ArrayStateSpace : public StateSpace {
+private:
+    std::vector<double> stateSpaceArray;
+    
+    int coords_to_index(const std::vector<int>& coords) const;
+
+public:
+    ArrayStateSpace(int dimensions, int dimensionSize, double initialValues = 0.0);
+
+    int get_dimensions() const override;
+
+    int get_dimension_size() const override;
+    
+    double get(const std::vector<int>& coords) const override;
+
+    std::vector<double> get_raw_representation() const;
+
+    double set(const std::vector<int>& coords, double value);
+    
+};
+
+#endif // ARRAY_STATE_SPACE_H
\ No newline at end of file
diff --git a/src/StateSpace/StateSpace.cpp b/src/StateSpace/StateSpace.cpp
deleted file mode 100644
index 267f8e3..0000000
--- a/src/StateSpace/StateSpace.cpp
+++ /dev/null
@@ -1,44 +0,0 @@
-#include "StateSpace.hpp"
-
-#include <string>
-
-int StateSpace::coords_to_index(const std::vector<int>& coords) const {
-    if (coords.size() != this->dimensions) 
-        throw std::invalid_argument("Index has Invalid Number of Dimensions");
-    int rawIndex = 0;
-    int multiplier = 1;
-    for (int i = dimensions - 1; i > -1; i--) {
-        if (coords[i] < 0 || coords[i] >= this->dimensionSize) 
-            throw std::out_of_range("Coordinate at index [" + std::to_string(i) + "] out of range");
-        rawIndex += coords[i] * multiplier;
-        multiplier *= dimensionSize;
-    }
-    return rawIndex;
-}
-
-StateSpace::StateSpace(int dimensions, int dimensionSize, double initialValues) 
-    :   dimensions(dimensions), 
-        dimensionSize(dimensionSize), 
-        stateSpace(std::pow(dimensionSize, dimensions), initialValues)
-    {}
-
-
-int StateSpace::get_dimensions() const {
-    return this->dimensions;
-}
-
-int StateSpace::get_dimension_size() const {
-    return this->dimensionSize;
-}
-
-double StateSpace::get(const std::vector<int>& coords) const {
-    return this->stateSpace[coords_to_index(coords)];
-}
-
-std::vector<double> StateSpace::get_raw_representation() const {
-    return this->stateSpace;
-}
-
-double StateSpace::set(const std::vector<int>& coords, double value) {
-    return this->stateSpace[coords_to_index(coords)] = value;
-}
\ No newline at end of file
diff --git a/src/StateSpace/StateSpace.hpp b/src/StateSpace/StateSpace.hpp
index 6928801..de5c045 100644
--- a/src/StateSpace/StateSpace.hpp
+++ b/src/StateSpace/StateSpace.hpp
@@ -6,33 +6,24 @@ Implements a StateSpace class that allows for interactions with an n-dimensional
 */
 
 
-#ifndef STATESPACE_H
-#define STATESPACE_H
+#ifndef STATE_SPACE_H
+#define STATE_SPACE_H
 #include <vector>
 #include <iostream>
 #include <cmath>
 
 class StateSpace {
-private:
-    std::vector<double> stateSpace;
-    const int dimensions;
-    const int dimensionSize;
-
-    int coords_to_index(const std::vector<int>& coords) const;
-
+protected:
+    int dimensions;
+    int dimensionSize;
 public:
-    StateSpace(int dimensions, int dimensionSize, double initialValues = 0.0);
-
-    int get_dimensions() const;
-
-    int get_dimension_size() const;
+    StateSpace(int dimensions, int dimensionSize): dimensions(dimensions), dimensionSize(dimensionSize) {}
     
-    double get(const std::vector<int>& coords) const;
-
-    std::vector<double> get_raw_representation() const;
+    virtual int get_dimensions() const = 0;
 
-    double set(const std::vector<int>& coords, double value);
+    virtual int get_dimension_size() const = 0;
     
+    virtual double get(const std::vector<int>& coords) const = 0;
 };
 
-#endif // STATESPACE_H
\ No newline at end of file
+#endif // STATE_SPACE_H
\ No newline at end of file
diff --git a/src/main.cpp b/src/main.cpp
index 1a668e9..31dd85b 100644
--- a/src/main.cpp
+++ b/src/main.cpp
@@ -1,52 +1,46 @@
 #include <vector>
 
 #include "InputOutput/CommandLineInputOutput.hpp"
-#include "Models/DumbModel.hpp"
-#include "Models/LinearModel.hpp"
 
-void linearAlgorithm(int dimensions, int dimensionSize, int queries){
+#if defined(LINEAR)
+    #include "Models/LinearModel.hpp"
+    #define CurrentModel LinearModel
+#elif defined(DUMB)
+    #include "Models/DumbModel.hpp"
+    #define CurrentModel DumbModel
+#elif defined(RBF)
+    #include "Models/RBFModel.hpp"
+    #define CurrentModel RBFModel
+#else
+    #error "Algorthim was not defined please check readme for build instructions"
+#endif
+
+void algorithm(int dimensions, int dimensionSize, int totalQueries){
     InputOutput *io = InputOutput::get_instance();
 
-    LinearModel model(dimensions, dimensionSize, queries);
+    CurrentModel model(dimensions, dimensionSize, totalQueries);
 
-    for (int i = 0; i < queries; i++){
+    for (int i = 0; i < totalQueries; i++){
         std::vector<int> query = model.get_next_query();
         double result = io->send_query_recieve_result(query);
         model.update_prediction(query, result);
     }
-    model.update_prediction_final();
-    const auto raw_state = model.get_state_space();
-    io->output_state(raw_state);
-}
-
-void dumbAlgorithm(int dimensions, int dimensionSize, int queries){
-    InputOutput *io = InputOutput::get_instance();
-
-    DumbModel model(dimensions, dimensionSize);
-
-    for (int i = 0; i < queries; i++){
-        std::vector<int> query = model.get_next_query();
-        double result = io->send_query_recieve_result(query);
-        model.update_prediction(query, result);
-    }
-    const auto raw_state = model.get_state_space();
-    io->output_state(raw_state);
+    io->output_state(model);
 }
 
 int main(int argc, char* argv[]) {
     if (argc != 4) { // program name + 3 integers
-        std::cerr << "Usage: " << argv[0] << " Dimensions : int,  Array size : int,  Maximum number of queries : int\n";
+        std::cerr << "Usage: " << argv[0] << " Dimensions : int,  Array size : int,  Maximum number of totalQueries : int\n";
         return 1;
     }
-
+    
     int dimensions = std::atoi(argv[1]);
     int dimensionSize = std::atoi(argv[2]);
-    int queries = std::atoi(argv[3]);
+    int totalQueries = std::atoi(argv[3]);
 
     CommandLineInputOutput::set_IO();
 
-    // dumbAlgorithm(dimensions, dimensionSize, queries);
-    linearAlgorithm(dimensions, dimensionSize, queries);
+    algorithm(dimensions, dimensionSize, totalQueries);
 
     return 0;
 }
diff --git a/src/main_test.txt b/src/main_test.txt
new file mode 100644
index 0000000..9f43b9d
--- /dev/null
+++ b/src/main_test.txt
@@ -0,0 +1,67 @@
+#include <vector>
+#include <iostream>  // <-- for logging
+
+#include "InputOutput/CommandLineInputOutput.hpp"
+
+#if defined(LINEAR)
+    #include "Models/LinearModel.hpp"
+    #define CurrentModel LinearModel
+#elif defined(DUMB)
+    #include "Models/DumbModel.hpp"
+    #define CurrentModel DumbModel
+#elif defined(RBF)
+    #include "Models/RBFModel.hpp"
+    #define CurrentModel RBFModel
+#else
+    #error "Algorthim was not defined please check readme for build instructions"
+#endif
+
+void algorithm(int dimensions, int dimensionSize, int totalQueries){
+    InputOutput *io = InputOutput::get_instance();
+
+    std::cerr << "[LOG] Initializing model..." << std::endl;
+    CurrentModel model(dimensions, dimensionSize, totalQueries);
+    std::cerr << "[LOG] Model initialized." << std::endl;
+
+    for (int i = 0; i < totalQueries; i++){
+        std::cerr << "[LOG] Iteration " << i+1 << "/" << totalQueries << ": Getting next query..." << std::endl;
+        std::vector<int> query = model.get_next_query();
+        std::cerr << "[LOG] Got query of size " << query.size() << std::endl;
+
+        std::cerr << "[LOG] Sending query to IO..." << std::endl;
+        double result = io->send_query_recieve_result(query);
+        std::cerr << "[LOG] Got result = " << result << std::endl;
+
+        std::cerr << "[LOG] Updating prediction..." << std::endl;
+        model.update_prediction(query, result);
+        std::cerr << "[LOG] Update done." << std::endl;
+    }
+
+    std::cerr << "[LOG] Outputting final state..." << std::endl;
+    io->output_state(model);
+    std::cerr << "[LOG] Finished algorithm." << std::endl;
+}
+
+int main(int argc, char* argv[]) {
+    if (argc != 4) { // program name + 3 integers
+        std::cerr << "Usage: " << argv[0] << " Dimensions : int,  Array size : int,  Maximum number of totalQueries : int\n";
+        return 1;
+    }
+    
+    int dimensions = std::atoi(argv[1]);
+    int dimensionSize = std::atoi(argv[2]);
+    int totalQueries = std::atoi(argv[3]);
+
+    std::cerr << "[LOG] Parsed arguments: dimensions=" 
+              << dimensions << ", dimensionSize=" 
+              << dimensionSize << ", totalQueries=" 
+              << totalQueries << std::endl;
+
+    CommandLineInputOutput::set_IO();
+    std::cerr << "[LOG] IO initialized." << std::endl;
+
+    algorithm(dimensions, dimensionSize, totalQueries);
+
+    std::cerr << "[LOG] Program finished successfully." << std::endl;
+    return 0;
+}
diff --git a/test/InputOutputTest.cpp b/test/InputOutputTest.cpp
index 1e20305..dad339d 100644
--- a/test/InputOutputTest.cpp
+++ b/test/InputOutputTest.cpp
@@ -4,7 +4,6 @@
 #include "../src/InputOutput/InputOutput.hpp"
 #include "../src/InputOutput/CommandLineInputOutput.hpp"
 
-using namespace std;
 
 TEST(TestCommandLine, testCLIOinnit){
     // use clio input output
diff --git a/test/ModelTest.cpp b/test/ModelTest.cpp
index c74bd0b..ecdfa02 100644
--- a/test/ModelTest.cpp
+++ b/test/ModelTest.cpp
@@ -3,42 +3,41 @@
 
 #include "../src/Models/DumbModel.hpp"
 
-using namespace std;
 
 TEST(TestModel, TestsModel1D){
-    DumbModel myModel(1, 10);
+    DumbModel myModel(1, 10, 10);
 
     // test functionality
     EXPECT_EQ(1, myModel.get_next_query().size());
-    EXPECT_EQ(0, myModel.get_state_space().get({1}));
+    EXPECT_EQ(0, myModel.get_value_at({1}));
     EXPECT_NO_THROW(myModel.update_prediction({1}, 1));
-    EXPECT_EQ(1, myModel.get_state_space().get({1}));
+    EXPECT_EQ(1, myModel.get_value_at({1}));
 
     // check that created array is correct size
-    EXPECT_NO_THROW(myModel.get_state_space().get({0}));
-    EXPECT_NO_THROW(myModel.get_state_space().get({9}));
-    EXPECT_THROW(myModel.get_state_space().get({10}), std::out_of_range);
+    EXPECT_NO_THROW(myModel.get_value_at({0}));
+    EXPECT_NO_THROW(myModel.get_value_at({9}));
+    EXPECT_THROW(myModel.get_value_at({10}), std::out_of_range);
 }
 
 TEST(TestModel, TestsModelND){
     for (int i = 1; i < 6; i++){
-        DumbModel myModel(i, 10);
+        DumbModel myModel(i, 10, 10);
         
-        vector<int> queryPoint(i, 0);
+        std::vector<int> queryPoint(i, 0);
 
         // test functionality
         EXPECT_EQ(i, myModel.get_next_query().size());
-        EXPECT_EQ(0, myModel.get_state_space().get(queryPoint));
+        EXPECT_EQ(0, myModel.get_value_at(queryPoint));
         EXPECT_NO_THROW(myModel.update_prediction(queryPoint, 1));
-        EXPECT_EQ(1, myModel.get_state_space().get(queryPoint));
+        EXPECT_EQ(1, myModel.get_value_at(queryPoint));
 
         // check that created array is correct size
         for (int j = 0; j < i; j++){
-            EXPECT_NO_THROW(myModel.get_state_space().get(queryPoint));
+            EXPECT_NO_THROW(myModel.get_value_at(queryPoint));
             queryPoint[j] = 9;
-            EXPECT_NO_THROW(myModel.get_state_space().get(queryPoint));
+            EXPECT_NO_THROW(myModel.get_value_at(queryPoint));
             queryPoint[j] = 10;
-            EXPECT_THROW(myModel.get_state_space().get(queryPoint), std::out_of_range);
+            EXPECT_THROW(myModel.get_value_at(queryPoint), std::out_of_range);
             queryPoint[j] = 9;
         }
     }
diff --git a/test/StateSpaceTest.cpp b/test/StateSpaceTest.cpp
index ec7836a..e0ff0d6 100644
--- a/test/StateSpaceTest.cpp
+++ b/test/StateSpaceTest.cpp
@@ -1,12 +1,10 @@
 #include <iostream>
 #include <gtest/gtest.h>
 
-#include "../src/StateSpace/StateSpace.hpp"
-
-using namespace std;
+#include "../src/StateSpace/ArrayStateSpace.hpp"
 
 TEST(TestStateSpace, TestsGetAndSetIn1D){
-    StateSpace mySpace(1, 5);
+    ArrayStateSpace mySpace(1, 5);
     ASSERT_EQ(0.0, mySpace.get({1}));
     mySpace.set({1}, -1);
     ASSERT_EQ(-1.0, mySpace.get({1}));
@@ -14,11 +12,11 @@ TEST(TestStateSpace, TestsGetAndSetIn1D){
 
 TEST(TestStateSpace, TestsGetAndSetInND){
     for (int i = 1; i < 6; i++){
-        StateSpace mySpace(i, 10);
+        ArrayStateSpace mySpace(i, 10);
         EXPECT_EQ(i, mySpace.get_dimensions());
         EXPECT_EQ(10, mySpace.get_dimension_size());
 
-        vector<int> point(i, i);
+        std::vector<int> point(i, i);
     
         EXPECT_EQ(0.0, mySpace.get(point));
         EXPECT_NO_THROW(mySpace.set(point, 23));
@@ -27,7 +25,7 @@ TEST(TestStateSpace, TestsGetAndSetInND){
 }
 
 TEST(TestStateSpace, TestsErrorThrowing){
-    StateSpace mySpace(1, 5);
+    ArrayStateSpace mySpace(1, 5);
 
     EXPECT_THROW(mySpace.get({6}), std::out_of_range);
 
@@ -36,7 +34,7 @@ TEST(TestStateSpace, TestsErrorThrowing){
     EXPECT_THROW(mySpace.get({5, 5}), std::invalid_argument);
 
 
-    StateSpace mySpace2(4, 4);
+    ArrayStateSpace mySpace2(4, 4);
     
     EXPECT_THROW(mySpace2.get({1,1,1,5}), std::out_of_range);
     EXPECT_THROW(mySpace2.set({1,1,1,5}, 1), std::out_of_range);
