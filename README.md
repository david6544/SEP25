# SEP25
COMP SCI - 3006 - Software Engineering Project

## Building the Project

This project uses CMake as its build system. The project structure includes:
- `/src` - Source code files
- `/test` - Unit tests using Google Test

### Prerequisites

- C++20 compatible compiler
- CMake 3.16 or higher

### Build Instructions

1. Create the build directory and navigate to it:
`mkdir -p build && cd build`
2. Generate the build files:
`cmake ..`
3. Build the project:
`cmake --build .`
   ```

### Running the Application

After building, you can run the main executable:
```bash
./build/sep25_main
```

### Running Tests

To run the unit tests:
```bash
./build/sep25_tests
```

Or use CTest for more detailed output:
```bash
cd build
ctest
```

### Development Workflow

For development, you can use the following commands:

1. Clean build:
   ```bash
   rm -rf build
   cmake -S . -B build
   cmake --build build
   ```

2. Run tests after building:
   ```bash
   cmake --build build && ./build/sep25_tests
   ```
