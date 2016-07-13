cmake_minimum_required(VERSION 2.8)

## Required software
find_package(OpenCV REQUIRED)

## Sources and headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})

## Find all source files to be compiled into object files
file(GLOB SOURCES *.cpp gco/*.cpp)

## C++ compiler options
set(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++14 -Wa,--gstabs -save-temps")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

## 1. Compile...
add_executable(tim ${SOURCES})

## 2. Link...
target_link_libraries(tim ${OpenCV_LIBS})

## 3. Install...
#install( TARGETS ${TARGET} RUNTIME DESTINATION bin )
file(GLOB JSON_FILES ${CMAKE_SOURCE_DIR}/data/*.json)

foreach(FILE ${JSON_FILES})
	add_custom_command(TARGET tim
		POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/data
		COMMAND ${CMAKE_COMMAND} -E copy ${FILE} ${CMAKE_CURRENT_BINARY_DIR}/data)
endforeach()

#file(COPY ${JSON_FILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data)
