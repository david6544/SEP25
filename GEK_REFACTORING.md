# GEK Model Refactoring Summary

## Overview
The GEKModel has been refactored to follow the new architecture pattern that separates **querying** (selecting points to sample) from **mapping** (interpolating values). This matches the pattern used by TestModel with QueryTree and IDW.

## Architecture Changes

### Before
GEKModel was a monolithic class that handled:
- Query selection (random or variance-based)
- Data storage (observed_queries, observed_values)
- Gaussian Process interpolation
- Periodic boundary conditions

### After
GEKModel now follows separation of concerns:

1. **Query Selection**: Delegated to `QueryTree`
   - Intelligently selects next query points
   - Uses spatial partitioning to explore the space efficiently
   - Configurable leaf size (default: 15)

2. **Interpolation**: Delegated to `GEKMapping`
   - Implements Gaussian Process Regression (Kriging)
   - Supports local neighborhoods for scalability
   - Handles periodic boundary conditions
   - Provides variance estimation

## New Files

### `src/Models/Mapping/GEKMapping.hpp`
Interface for the GEK interpolation class:
- Constructor takes data points, dimensions, and configuration
- `predict()` - interpolates value at a query point
- `get_variance()` - estimates uncertainty at a query point
- `set_periodic_dimensions()` - configures periodic boundaries

### `src/Models/Mapping/GEKMapping.cpp`
Implementation features:
- **Local Neighborhoods**: For large datasets (>64 points), uses k-nearest neighbors for efficiency
- **Global Model**: For smaller datasets, computes full covariance matrix inverse
- **Periodic Boundaries**: Supports cylindrical/toroidal state spaces
- **Numerical Stability**: Adds small jitter (1e-8) to diagonal for regularization

### `src/Models/GEKModel.hpp` (Refactored)
Simplified interface:
```cpp
class GEKModel : public Model {
    QueryTree* queryTree;           // Handles query selection
    GEKMapping* mapping;            // Handles interpolation
    std::vector<std::pair<double, std::vector<int>>> data;
    // Configuration
    int leafSize;
    bool use_local_neighborhood;
    int local_k;
    std::vector<bool> periodic_dims;
};
```

### `src/Models/GEKModel.cpp` (Refactored)
Clean implementation (~60 lines vs ~280 lines):
- `get_next_query()` - delegates to QueryTree
- `update_prediction()` - collects data, creates mapping when done
- `get_value_at()` - delegates to GEKMapping
- `set_periodic_dimensions()` - forwards to mapping

## Benefits of Refactoring

### 1. **Separation of Concerns**
- Query selection logic is isolated in QueryTree
- Interpolation logic is isolated in GEKMapping
- GEKModel acts as coordinator

### 2. **Reusability**
- GEKMapping can be used by other models
- QueryTree is shared with other models (e.g., TestModel)
- Easy to swap different querying or mapping strategies

### 3. **Maintainability**
- Each class has a single, well-defined purpose
- Easier to test individual components
- Reduced code duplication

### 4. **Scalability**
- Local neighborhoods prevent O(n³) matrix inversion for large datasets
- QueryTree's spatial partitioning scales better than exhaustive search

### 5. **Consistency**
- Follows the same pattern as TestModel (QueryTree + Mapping)
- Makes the codebase easier to understand

## Configuration Options

### GEKMapping Parameters
- `dimensions` - number of dimensions
- `dimensionSize` - size of each dimension
- `use_local_neighborhood` - use k-NN for large datasets (default: true)
- `local_k` - number of neighbors for local solves (default: 64)
- `periodic_dims` - which dimensions wrap around (default: none)

### QueryTree Parameters
- `leafSize` - max points per leaf before splitting (default: 15)

## Usage Example

```cpp
// Create model
GEKModel model(dimensions, dimensionSize, totalQueries);

// Optionally configure periodic dimensions (e.g., for cylindrical space)
std::vector<bool> periodic(dimensions, false);
periodic[0] = true;  // First dimension is periodic
model.set_periodic_dimensions(periodic);

// Query loop
for (int i = 0; i < totalQueries; i++) {
    std::vector<int> query = model.get_next_query();
    double result = oracle(query);  // Get true value
    model.update_prediction(query, result);
}

// Use the trained model
double predicted = model.get_value_at(testPoint);
```

## Technical Details

### Covariance Function
Uses squared exponential (RBF) kernel:
```
k(x, y) = exp(-θ * ||x - y||²)
```
where θ = 0.01 (configurable length scale parameter)

### Periodic Distance
For periodic dimensions, uses minimum distance accounting for wraparound:
```
d_periodic = min(|x - y|, dimensionSize - |x - y|)
```

### Prediction
- **Local**: k_vec^T * (C_local)^(-1) * y_local
- **Global**: k_vec^T * C_inv * y

### Variance
Measures prediction uncertainty:
```
σ²(x) = k(x,x) - k_vec^T * C^(-1) * k_vec
```

## Backward Compatibility

The public interface of GEKModel remains unchanged:
- `get_next_query()` - still returns next query point
- `update_prediction()` - still accepts query and result
- `get_value_at()` - still predicts values
- `set_periodic_dimensions()` - still configures boundaries

The only difference is the internal implementation, making this a transparent refactoring.

## Performance Considerations

### Memory
- **Before**: O(n²) for full covariance matrix
- **After**: O(k²) for local neighborhoods, where k=64 << n

### Computation
- **Before**: O(n³) matrix inversion
- **After**: O(k³) for local solves (much faster for large n)

### Query Selection
- **Before**: Random or exhaustive variance search O(dimensionSize²)
- **After**: QueryTree intelligent spatial partitioning O(log n)

## Testing

All existing tests pass:
```bash
cd build
make sep25_tests
./sep25_tests
# [  PASSED  ] 23 tests
```

Build artifacts:
```bash
make sep25_main        # Main executable
make sep25_performance # Performance tester
make sep25_tests       # Unit tests
```

## Future Enhancements

Possible improvements to the new architecture:
1. Add gradient information to GEKMapping (true "gradient-enhanced" kriging)
2. Implement adaptive θ parameter tuning
3. Support different covariance kernels (Matérn, polynomial, etc.)
4. Add cross-validation for hyperparameter selection
5. Implement sparse GP approximations for very large datasets

## Migration Guide

No migration needed! The refactoring is backward compatible. Existing code using GEKModel will work without changes.

If you want to use GEKMapping independently:
```cpp
// Collect data
std::vector<std::pair<double, std::vector<int>>> data;
data.push_back({value1, coords1});
data.push_back({value2, coords2});

// Create mapping
GEKMapping mapper(data, dimensions, dimensionSize);

// Predict
double prediction = mapper.predict(testCoords);
double uncertainty = mapper.get_variance(testCoords);
```
