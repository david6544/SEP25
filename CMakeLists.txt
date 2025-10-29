cmake_minimum_required(VERSION 3.16)
project(SEP25 VERSION 0.1.0 LANGUAGES CXX)

# Selected Algorithm (change text in " ")
set(ALGORITHM "DUMB")

# Detect if building natively or with a toolchain
if(CMAKE_TOOLCHAIN_FILE)
    message(STATUS "Using toolchain: ${CMAKE_TOOLCHAIN_FILE}")
else()
    message(STATUS "Building natively on ${CMAKE_SYSTEM_NAME}")
endif()

# Prevent reusing a cross-compilation build dir for native builds
if(NOT CMAKE_TOOLCHAIN_FILE AND CMAKE_CROSSCOMPILING)
    message(FATAL_ERROR "This build directory was configured for cross-compilation. "
                        "Use a fresh build directory for native builds.")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# --- GIT CHECKS (host side) ---
if(CMAKE_CROSSCOMPILING)
    # --- GIT CHECKS (only for cross-compilation) ---
    find_package(Git QUIET)
    if(NOT Git_FOUND)
        message(FATAL_ERROR "Git not found, cannot verify commit status.")
    endif()

    # Ensure repository has at least one commit
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --verify HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        RESULT_VARIABLE GIT_HEAD_RESULT
    )
    if(NOT GIT_HEAD_RESULT EQUAL 0)
        message(FATAL_ERROR "No commits found in repository. Please commit before building.")
    endif()

    # Ensure working tree is clean
    execute_process(
        COMMAND ${GIT_EXECUTABLE} status --porcelain
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_STATUS
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT "${GIT_STATUS}" STREQUAL "")
        message(FATAL_ERROR "Uncommitted changes detected. Commit or stash all changes before building.")
    endif()

    # Get commit hash
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_COMMIT_HASH_SHORT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    message(STATUS "Cross-compiling from clean commit: ${GIT_COMMIT_HASH_SHORT}")
else()
    set(GIT_COMMIT_HASH_SHORT "dev") # Default name for native builds
    message(STATUS "Native build (no Git enforcement)")
endif()

# --- SOURCES ---
file(GLOB_RECURSE SRC_FILES src/*.cpp)
list(FILTER SRC_FILES EXCLUDE REGEX ".*/main.cpp$")

# Main executable with commit hash
add_executable(sep25_main src/main.cpp ${SRC_FILES})

message(STATUS "Selected algorithm: ${ALGORITHM}")

# Add compile definition based on chosen algorithm
target_compile_definitions(sep25_main PUBLIC ${ALGORITHM})

target_include_directories(sep25_main PUBLIC ${CMAKE_BINARY_DIR})
include_directories(SYSTEM ${CMAKE_CXX_STANDARD_LIBRARIES})
include_directories(${eigen_SOURCE_DIR})
set_target_properties(sep25_main PROPERTIES OUTPUT_NAME "sep25_main_${GIT_COMMIT_HASH_SHORT}")

# Add performance test executable
file(GLOB PERFORMANCE_FILES performanceTester/*.cpp)
add_executable(sep25_performance ${PERFORMANCE_FILES} ${SRC_FILES})
target_include_directories(sep25_performance PUBLIC ${CMAKE_BINARY_DIR})
target_compile_definitions(sep25_performance PUBLIC ${ALGORITHM})
include_directories(${eigen_SOURCE_DIR})

# GoogleTest setup (unchanged)
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.17.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Eigen for linear algebra
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)

enable_testing()
file(GLOB TEST_FILES test/*.cpp)
add_executable(sep25_tests ${SRC_FILES} ${TEST_FILES})
target_compile_definitions(sep25_tests PUBLIC "TESTING")
include_directories(${eigen_SOURCE_DIR})

# Link GoogleTest
target_link_libraries(sep25_tests 
    gtest_main
    gmock_main
)

# Add tests
target_link_libraries(sep25_tests gtest_main gmock_main)
add_test(NAME UnitTests COMMAND sep25_tests)

# MinGW static linking for cross-build
if(MINGW)
    message(STATUS "Building with MinGW, enabling static linking")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")
endif()

# --- DOXYGEN DOCUMENTATION ---
find_package(Doxygen QUIET)

if(DOXYGEN_FOUND)
    # Canonical Doxyfile is stored in the repo root
    set(DOXYGEN_IN ${CMAKE_SOURCE_DIR}/Doxyfile)

    # Developers can make their own local ./docs directory for output
    # Output is inside the current build dir unless overridden in Doxyfile
    set(DOXYGEN_OUT ${CMAKE_BINARY_DIR}/docs)

    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_IN}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
else()
    message(WARNING "Doxygen not found. 'docs' target will be unavailable.")
endif()


