# GEK Model Architecture Diagrams

## Before Refactoring (Monolithic)

```
┌─────────────────────────────────────────────────────┐
│              GEKModel                                │
│                                                      │
│  ┌────────────────────────────────────────────┐    │
│  │         Query Selection Logic               │    │
│  │  • get_random_query()                       │    │
│  │  • get_variance() for 2D exhaustive search  │    │
│  │  • Deduplication tracking                   │    │
│  │  • Random number generator                  │    │
│  └────────────────────────────────────────────┘    │
│                                                      │
│  ┌────────────────────────────────────────────┐    │
│  │         Data Storage                        │    │
│  │  • observed_queries (vector<vector<int>>)   │    │
│  │  • observed_values (vector<double>)         │    │
│  │  • stateSpace (ArrayStateSpace)             │    │
│  └────────────────────────────────────────────┘    │
│                                                      │
│  ┌────────────────────────────────────────────┐    │
│  │      Gaussian Process Interpolation         │    │
│  │  • covariance()                             │    │
│  │  • compute_periodic_distance_squared()      │    │
│  │  • train()                                  │    │
│  │  • C_inv (global covariance inverse)        │    │
│  │  • Local neighborhood logic                 │    │
│  └────────────────────────────────────────────┘    │
│                                                      │
└─────────────────────────────────────────────────────┘
        ~280 lines of tightly coupled code
```

## After Refactoring (Modular)

```
┌──────────────────────────────────────┐
│         GEKModel                      │
│      (Coordinator)                    │
│                                       │
│  • get_next_query()                  │
│  • update_prediction()               │
│  • get_value_at()                    │
│  • set_periodic_dimensions()         │
│                                       │
│  ~60 lines                            │
└───────────┬──────────────┬───────────┘
            │              │
    delegates to     delegates to
            │              │
            ▼              ▼
    ┌───────────┐  ┌──────────────┐
    │ QueryTree │  │ GEKMapping   │
    └───────────┘  └──────────────┘

┌─────────────────────────────────────────────┐
│              QueryTree                       │
│         (Query Selection)                    │
│                                              │
│  ┌────────────────────────────────────┐    │
│  │   Spatial Partitioning Tree         │    │
│  │  • Intelligent point selection      │    │
│  │  • Adaptive space subdivision       │    │
│  │  • Variance-guided exploration      │    │
│  └────────────────────────────────────┘    │
│                                              │
│  Used by: TestModel, GEKModel               │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│            GEKMapping                        │
│    (GP Interpolation - Mapping subclass)    │
│                                              │
│  ┌────────────────────────────────────┐    │
│  │   Gaussian Process Regression       │    │
│  │  • predict()                        │    │
│  │  • get_variance()                   │    │
│  │  • covariance()                     │    │
│  │  • Local neighborhoods (k-NN)       │    │
│  │  • Global model (full inverse)      │    │
│  │  • Periodic boundary support        │    │
│  └────────────────────────────────────┘    │
│                                              │
│  Implements: Mapping interface              │
│  ~250 lines                                  │
└─────────────────────────────────────────────┘
```

## Component Interaction Flow

### Query Phase
```
User Code
   │
   ├─► GEKModel.get_next_query()
   │       │
   │       └─► QueryTree.get_next_query()
   │               │
   │               ├─► Evaluate candidates
   │               ├─► Score based on data density
   │               └─► Return best point
   │
   └─► Returns query to user
```

### Update Phase
```
User Code (has query result)
   │
   ├─► GEKModel.update_prediction(query, result)
   │       │
   │       ├─► QueryTree.update_prediction(query, result)
   │       │       │
   │       │       ├─► Add point to tree
   │       │       └─► Trigger split if needed
   │       │
   │       ├─► Store in data vector
   │       │
   │       └─► If last query:
   │               │
   │               └─► Create GEKMapping(data, ...)
   │                       │
   │                       └─► Optional: train global model
   │
   └─► Done
```

### Prediction Phase
```
User Code
   │
   ├─► GEKModel.get_value_at(point)
   │       │
   │       └─► GEKMapping.predict(point)
   │               │
   │               ├─► Check if exact match
   │               │
   │               ├─► If many points: use local k-NN
   │               │       │
   │               │       ├─► Select k nearest neighbors
   │               │       ├─► Build local covariance matrix
   │               │       └─► Solve local GP
   │               │
   │               └─► Else: use global model
   │                       │
   │                       └─► k_vec^T * C_inv * y
   │
   └─► Returns prediction
```

## Dependency Graph

```
                    ┌──────────┐
                    │  Model   │ (abstract)
                    └─────┬────┘
                          │
                    inherits from
                          │
                          ▼
              ┌───────────────────┐
              │    GEKModel        │
              └───────┬───────┬───┘
                      │       │
                   uses    uses
                      │       │
         ┌────────────┘       └────────────┐
         ▼                                  ▼
    ┌──────────┐                    ┌──────────┐
    │QueryTree │                    │GEKMapping│
    └──────┬───┘                    └────┬─────┘
           │                              │
        uses                          inherits from
           │                              │
           ▼                              ▼
    ┌─────────────────┐            ┌──────────┐
    │CandidateGenerator│            │ Mapping  │ (abstract)
    └─────────────────┘            └──────────┘
```

## File Organization

```
src/Models/
├── Model.hpp                   # Abstract base class
├── GEKModel.hpp               # Refactored coordinator
├── GEKModel.cpp               # ~60 lines
│
├── Querying/
│   ├── QueryTree.hpp          # Spatial partitioning
│   └── QueryTree.cpp          # Shared with TestModel
│
├── Mapping/
│   ├── Mapping.hpp            # Abstract interpolator
│   ├── GEKMapping.hpp         # GP interpolation
│   ├── GEKMapping.cpp         # ~250 lines
│   ├── IDW.hpp                # Alternative: Inverse Distance
│   └── IDW.cpp                # Used by TestModel
│
└── Tools/
    └── CandidateGenerator.hpp # Used by QueryTree
```

## Data Flow

### Training Data Collection
```
Iteration 1:
  Query → Result → QueryTree + data vector

Iteration 2:
  Query → Result → QueryTree + data vector

...

Iteration N (final):
  Query → Result → QueryTree + data vector
                       ↓
               Create GEKMapping(data)
                       ↓
                 Train if needed
```

### Prediction (after training)
```
Query Point
     ↓
GEKModel.get_value_at()
     ↓
GEKMapping.predict()
     ↓
┌─────────────────┐
│ If n > local_k: │
│  Use k-NN       │
│ Else:           │
│  Use global     │
└─────────────────┘
     ↓
 Prediction
```

## Class Responsibilities

| Class | Responsibility | Lines | Reusable |
|-------|---------------|-------|----------|
| `GEKModel` | Coordinate query & mapping | 60 | ✗ |
| `QueryTree` | Select next query intelligently | ~200 | ✓ |
| `GEKMapping` | GP interpolation | 250 | ✓ |
| `Mapping` | Define interpolation interface | 10 | ✓ |
| `Model` | Define model interface | 25 | ✓ |

## Comparison with TestModel

Both models now follow the same pattern:

```
TestModel:                     GEKModel:
┌──────────────┐              ┌──────────────┐
│  TestModel   │              │  GEKModel    │
└──┬────────┬──┘              └──┬────────┬──┘
   │        │                    │        │
   ▼        ▼                    ▼        ▼
QueryTree  IDW              QueryTree  GEKMapping
(same!)  (simple)           (same!)  (sophisticated)
```

**Key Insight**: Both models share QueryTree for query selection, but use different mapping strategies (IDW vs GEK).

## Extension Points

The new architecture makes it easy to:

### 1. Add New Query Strategies
```cpp
class UCBQuerySelector : public QuerySelector {
    std::vector<int> get_next_query() override {
        // Upper Confidence Bound selection
    }
};

GEKModel model(...);
model.setQueryStrategy(new UCBQuerySelector());
```

### 2. Add New Mapping Strategies
```cpp
class NeuralNetMapping : public Mapping {
    double predict(std::vector<int> query) override {
        // Neural network interpolation
    }
};

GEKModel model(...);
model.setMapping(new NeuralNetMapping());
```

### 3. Mix and Match
```cpp
// GEK interpolation with UCB query selection
GEKModel model1(dims, dimSize, queries);
model1.setQueryStrategy(new UCBQuerySelector());

// IDW interpolation with QueryTree selection
TestModel model2(dims, dimSize, queries);
// Already uses QueryTree + IDW
```

## Summary

The refactoring transforms a monolithic 325-line GEKModel into a modular system with clear responsibilities:
- **GEKModel**: 60-line coordinator
- **QueryTree**: Reusable query selector
- **GEKMapping**: Reusable GP interpolator

This follows the Single Responsibility Principle and makes the codebase more maintainable, testable, and extensible.
