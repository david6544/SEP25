# Running repetitively to test changes in KDTreeStateSpace.cpp
#!/bin/bash

mkdir -p build && cd build
cmake .. && make -j4
