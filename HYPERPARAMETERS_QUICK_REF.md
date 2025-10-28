# RegionModel Hyperparameters - Quick Reference Card

## 🚀 Quick Start

```cpp
#include "Models/RegionModelPresets.hpp"

// Recommended for 100³ with 300k queries
auto config = RegionModelPresets::Optimized100Cube();
RegionModel model(3, 100, 300000, config);
```

## 📊 Available Presets

| Preset | Use Case | Initial Queries | Total Queries | Best For |
|--------|----------|----------------|---------------|----------|
| `Default()` | Balanced, conservative | ~5k (5%) | ~250k | Unknown functions |
| `Optimized100Cube()` ⭐ | **100³ grid optimized** | **~8k (2.7%)** | **~200k** | **General use** |
| `BudgetSaver()` | Minimal queries | ~2k (1.7%) | ~120k | Query-constrained |
| `HighAccuracy()` | Maximum precision | ~17k (5.8%) | ~450k | Larger budgets |
| `SmoothFunctions()` | Continuous functions | ~3k (2.7%) | ~150k | Bowl/Plate/Valley |
| `DiscontinuousFunctions()` | Many boundaries | ~10k (3.3%) | ~280k | Rastrigin/Ackley |

## 🎛️ Key Parameters to Tune

### Most Impactful (Tune These First)

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `initial_grid_step` | 10 | 7-12 | **High** - Controls initial coverage |
| `variance_threshold_discontinuity` | 0.15 | 0.10-0.20 | **High** - Boundary detection |
| `knn_k` | 8 | 5-15 | **Medium** - Prediction smoothness |
| `samples_boundary_walking` | 8 | 6-12 | **Medium** - Boundary accuracy |

### Fine-Tuning Parameters

| Parameter | Default | Impact |
|-----------|---------|--------|
| `gradient_threshold` | 0.3 | Boundary sensitivity |
| `samples_initial_grid` | 5 | Initial phase speed |
| `samples_region_high_var` | 10 | Uncertainty handling |
| `neighbor_search_radius` | 2 | Boundary expansion |

## 🎯 Common Tuning Scenarios

### Scenario 1: Running Out of Queries
```cpp
config.initial_grid_step = 12;        // Sparser grid
config.samples_initial_grid = 3;      // Fewer samples
config.samples_region_low_var = 2;    // Minimal in smooth areas
```

### Scenario 2: Missing Discontinuities
```cpp
config.variance_threshold_discontinuity = 0.12;  // More sensitive
config.gradient_threshold = 0.25;                 // Lower threshold
config.samples_boundary_walking = 10;            // More confidence
```

### Scenario 3: Blocky/Noisy Predictions
```cpp
config.knn_k = 15;                    // More neighbors
config.samples_region_high_var = 12;  // More samples in uncertain areas
```

### Scenario 4: Need Faster Runtime
```cpp
config.initial_grid_step = 12;        // Fewer points
config.samples_initial_grid = 3;      // Quick sampling
config.min_samples_for_discontinuity = 2;  // Early detection
```

## 📈 Expected Performance (100³ grid, 300k budget)

### Optimized100Cube Configuration

| Function Type | MAE | RMSE | Queries Used |
|---------------|-----|------|--------------|
| Bowl (SumPow, HyperEllipsoid) | 0.05-0.10 | 0.10-0.15 | 180k-220k |
| Plate (PowerSum, Zakharov) | 0.06-0.09 | 0.08-0.12 | 160k-200k |
| Valley (Rosenbrock, DixonPrice) | 0.10-0.15 | 0.15-0.22 | 200k-240k |
| Many Minima (Rastrigin, Ackley, Griewank) | 0.15-0.25 | 0.20-0.30 | 220k-270k |
| Steep Ridges (Michalewicz) | 0.15-0.30 | 0.20-0.35 | 240k-280k |

## 🔧 Custom Configuration Template

```cpp
RegionModelConfig custom;

// STEP 1: Set coverage (affects initial queries)
custom.initial_grid_step = 8;  // 7=dense, 10=balanced, 12=sparse

// STEP 2: Set sample counts (affects total queries)
custom.samples_initial_grid = 4;        // Quick exploration
custom.samples_boundary_walking = 8;    // Boundary confidence
custom.samples_region_low_var = 3;      // Smooth regions
custom.samples_region_high_var = 10;    // Uncertain regions

// STEP 3: Set detection sensitivity (affects accuracy)
custom.variance_threshold_discontinuity = 0.15;  // Lower = more sensitive
custom.gradient_threshold = 0.30;                // Lower = more boundaries

// STEP 4: Set prediction quality (affects smoothness)
custom.knn_k = 10;  // Higher = smoother

RegionModel model(3, 100, 300000, custom);
```

## 🧪 Testing Workflow

```bash
# 1. Test on 20³ grid (quick validation)
./build/sep25_performance 3 20 --output true --rand false

# 2. Test on 30³ grid (check scaling)
./build/sep25_performance 3 30 --output true --rand false

# 3. Test on 50³ grid (realistic test)
./build/sep25_performance 3 50 --output true --rand false

# 4. Run full 100³ grid
./build/sep25_performance 3 100 --output true --rand false
```

## 🎓 Decision Tree

```
Is your function smooth (bowl/plate/valley)?
├─ YES → Use SmoothFunctions() preset
│         OR reduce sample counts by 2
│
└─ NO → Does it have many discontinuities (Rastrigin, Ackley)?
    ├─ YES → Use DiscontinuousFunctions() preset
    │         OR increase boundary_walking samples
    │
    └─ UNKNOWN → Use Optimized100Cube() preset
                  Start here and tune based on results
```

## 📝 Parameter Cheat Sheet

**Want to INCREASE accuracy?**
- ↑ `samples_boundary_walking`
- ↑ `knn_k`
- ↓ `variance_threshold_discontinuity`
- ↓ `gradient_threshold`

**Want to DECREASE query usage?**
- ↑ `initial_grid_step`
- ↓ `samples_initial_grid`
- ↓ `samples_region_low_var`
- ↑ `variance_threshold_discontinuity`

**Want SMOOTHER predictions?**
- ↑ `knn_k` (12-15)
- ↑ `samples_region_high_var`

**Want MORE DETAIL (less smooth)?**
- ↓ `knn_k` (5-7)
- ↓ `gradient_threshold`
- ↑ `neighbor_search_radius`

## 💡 Pro Tips

1. **Always test on 20³ first** - saves time and reveals issues early
2. **Start with Optimized100Cube** - it's tuned for your exact use case
3. **Monitor query usage** - add logging to track where queries go
4. **Variance thresholds are key** - they control everything
5. **KNN k affects visualization** - higher = prettier surface plots
6. **Budget 60-70% for boundary/filling** - don't spend all on initial grid

## 📚 Full Documentation

- **Complete Guide**: `docs/REGION_HYPERPARAMETERS.md`
- **Implementation Details**: `docs/HYPERPARAMETER_IMPLEMENTATION.md`
- **Code Examples**: `examples/hyperparameter_example.cpp`
- **Presets Source**: `src/Models/RegionModelPresets.hpp`

---

**TL;DR**: Use `RegionModelPresets::Optimized100Cube()` for the 100³ challenge with 300k queries. It's already tuned for you!
