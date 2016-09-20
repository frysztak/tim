cmake_minimum_required(VERSION 3.1)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(OpenCV REQUIRED core imgproc video videoio highgui)
find_package(PkgConfig REQUIRED)
pkg_search_module(nanomsg REQUIRED nanomsg)
find_package(Threads REQUIRED)

## Sources and headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})

## Export compile commands for YouCompleteMe to use
set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")

## Custom options
option(SIMD "Use SIMD optimizations. It only affects background substraction code." ON)
if (SIMD)
	add_definitions(-DSIMD)
endif()

option(PROFILE "Compile with flags that make it possible to profile the app with Callgrind." OFF)
if (PROFILE)
	set(CMAKE_BUILD_TYPE Release)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wa,--gstabs -save-temps")
endif()

option(DEBUG "Print some text and show some images that might be useful in debugging the algorithms.
              It doesn't force Debug build type." OFF)
if (DEBUG)
    add_definitions(-DDEBUG)
endif()

## Find all source files to be compiled into object files
file(GLOB SOURCES *.cpp)

## C++ compiler options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -std=c++14")

## Compile
add_executable(tim ${SOURCES})

## Link
target_link_libraries(tim ${OpenCV_LIBS} ${nanomsg_LIBRARIES} Threads::Threads)
