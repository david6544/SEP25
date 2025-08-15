cmake_minimum_required(VERSION 3.16)

project(SEP25 VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add src directory
file(GLOB SRC_FILES src/*.cpp)
add_executable(sep25_main ${SRC_FILES})
target_include_directories(sep25_main PUBLIC ${CMAKE_BINARY_DIR})
include_directories(SYSTEM ${CMAKE_CXX_STANDARD_LIBRARIES})


# Download and configure GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.17.0  # Use latest stable release
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Make GoogleTest available
FetchContent_MakeAvailable(googletest)

enable_testing()

# Create test executable
file(GLOB TEST_FILES test/*.cpp)
add_executable(sep25_tests ${TEST_FILES})

# Link GoogleTest
target_link_libraries(sep25_tests 
    gtest_main
    gmock_main
)

# Add tests
add_test(NAME UnitTests COMMAND sep25_tests)

