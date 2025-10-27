#!/bin/bash

# Test script for RegionModel optimizations
# Tests on progressively larger grids: 20³, 30³, 50³

echo "=================================================="
echo "Testing RegionModel Optimizations"
echo "=================================================="
echo ""

# Build the project
echo "Building project with REGION model..."
cmake -DALGORITHM=REGION -B build -S . > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ CMake configuration failed"
    exit 1
fi

cmake --build build > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Build failed"
    exit 1
fi
echo "✅ Build successful"
echo ""

# Test 1: 20³ grid (8,000 points, use ~2,400 queries = 30%)
echo "=================================================="
echo "Test 1: 20x20x20 grid (8,000 points)"
echo "Query budget: 2,400 queries (30%)"
echo "=================================================="
./build/sep25_performance 3 20 true false
if [ $? -eq 0 ]; then
    echo "✅ 20³ test completed"
else
    echo "⚠️  20³ test had issues"
fi
echo ""

# Test 2: 30³ grid (27,000 points, use ~8,100 queries = 30%)
echo "=================================================="
echo "Test 2: 30x30x30 grid (27,000 points)"
echo "Query budget: 8,100 queries (30%)"
echo "=================================================="
./build/sep25_performance 3 30 true false
if [ $? -eq 0 ]; then
    echo "✅ 30³ test completed"
else
    echo "⚠️  30³ test had issues"
fi
echo ""

# Test 3: 50³ grid (125,000 points, use ~37,500 queries = 30%)
echo "=================================================="
echo "Test 3: 50x50x50 grid (125,000 points)"
echo "Query budget: 37,500 queries (30%)"
echo "=================================================="
./build/sep25_performance 3 50 true false
if [ $? -eq 0 ]; then
    echo "✅ 50³ test completed"
else
    echo "⚠️  50³ test had issues"
fi
echo ""

# Summary
echo "=================================================="
echo "Testing Complete!"
echo "=================================================="
echo "Check the PerformanceData folders for results"
echo "Use the visualizer to inspect predictions:"
echo "  cd performanceTester/visualisation"
echo "  python3 -m main PerformanceData3:20 true"
echo "  python3 -m main PerformanceData3:30 true"
echo "  python3 -m main PerformanceData3:50 true"
echo ""
