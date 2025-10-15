# windows-toolchain.cmake
set(CMAKE_SYSTEM_NAME Windows)

# Detect host system to select correct mingw-w64 path
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
    # macOS with Homebrew mingw-w64
    set(MINGW_PREFIX /opt/homebrew/opt/mingw-w64/bin/x86_64-w64-mingw32)
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    # Linux with distro mingw-w64
    set(MINGW_PREFIX /usr/bin/x86_64-w64-mingw32)
else()
    message(FATAL_ERROR "Unsupported host system for Windows cross-compilation: ${CMAKE_HOST_SYSTEM_NAME}")
endif()

set(CMAKE_C_COMPILER   ${MINGW_PREFIX}-gcc)
set(CMAKE_CXX_COMPILER ${MINGW_PREFIX}-g++)

# Root path (for libraries + headers)
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
    set(CMAKE_FIND_ROOT_PATH /opt/homebrew/opt/mingw-w64/x86_64-w64-mingw32)
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    set(CMAKE_FIND_ROOT_PATH /usr/x86_64-w64-mingw32)
endif()

# Search rules for cross compilation
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
