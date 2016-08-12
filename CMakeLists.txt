cmake_minimum_required(VERSION 3.1)

## Required software
include(ExternalProject)
find_package(Git REQUIRED)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_search_module(nanomsg REQUIRED nanomsg)
find_package(Threads REQUIRED)

## Sources and headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})

## Catch
ExternalProject_Add(
	catch
	PREFIX ${CMAKE_BINARY_DIR}/catch
	GIT_REPOSITORY https://github.com/philsquared/Catch.git
	TIMEOUT 10
	UPDATE_COMMAND ${GIT_EXECUTABLE} pull
	
	CONFIGURE_COMMAND ""
	BUILD_COMMAND ""
	INSTALL_COMMAND ""
	LOG_DOWNLOAD ON)

# Expose required variable (CATCH_INCLUDE_DIR) to parent scope
ExternalProject_Get_Property(catch source_dir)
set(CATCH_INCLUDE_DIR ${source_dir}/include CACHE INTERNAL "Path to include folder for Catch")
set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")

# custom options
option(SIMD "Use SIMD optimizations" ON)
if (SIMD)
	add_definitions(-DSIMD)
endif()

option(PROFILE "Add flags that make it possible to profile the app with Callgrind" OFF)
if (PROFILE)
	set(CMAKE_BUILD_TYPE Release)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wa,--gstabs -save-temps")
endif()

## Find all source files to be compiled into object files
file(GLOB SOURCES *.cpp)

## C++ compiler options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -std=c++14")
set_source_files_properties(main.cpp PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)

#add_definitions(-DDEBUG)
## 1. Compile...
add_executable(tim ${SOURCES})

## 2. Link...
target_link_libraries(tim ${OpenCV_LIBS} ${nanomsg_LIBRARIES} Threads::Threads)

#enable_testing()
#add_subdirectory(tests)
