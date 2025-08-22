# SEP25
COMP SCI - 3006 - Software Engineering Project

## Building the Project

This project uses CMake as its build system. The project structure includes:
- `/src` - Source code files
- `/test` - Unit tests using Google Test

### Prerequisites

- C++20 compatible compiler
- CMake 3.16 or higher
- MinGW ( see below instructions)

### MinGW Installation
1. Install MinGW:
```bash
sudo apt update
sudo apt install mingw-w64
```

2. Validate Installation:
```bash
which x86_64-w64-mingw32-gcc 
which x86_64-w64-mingw32-g++
```


### Local Build Instructions

1. Create the build directory and navigate to it:
`mkdir -p build && cd build`
2. Generate the build files:
`cmake ..`
3. Build the project:
`cmake --build .`

### Windows Cross-Compilation Instructions
1. Create build folder `mkdir build-windows`
2. Generate the build files `cmake -B build-win -S . -DCMAKE_TOOLCHAIN_FILE=windows-toolchain.cmake -DCMAKE_BUILD_TYPE=Release &&  cmake --build build-windows`
4. .exe is located in `./build-windows/sep25_main_#######.exe` this cannot be run locally on a linux machine
can be copied to a windows machine for testing.


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
