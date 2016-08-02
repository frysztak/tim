cmake_minimum_required(VERSION 2.8)

## Required software
find_package(OpenCV REQUIRED)

## Sources and headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})

## Find all source files to be compiled into object files
file(GLOB SOURCES *.cpp)

## Data files
file(GLOB JSON_FILES ${CMAKE_SOURCE_DIR}/data/*.json)
add_custom_target(data DEPENDS ${JSON_FILES})

# Post-Build Step: Create required directory
add_custom_command(TARGET data POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E make_directory
	${CMAKE_CURRENT_BINARY_DIR}/data
	#COMMENT "Creating directory: " ${CMAKE_CURRENT_BINARY_DIR}/data
)

foreach(FILE ${JSON_FILES})
	add_custom_command(TARGET data POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
		${FILE}
		${CMAKE_CURRENT_BINARY_DIR}/data
		#COMMENT "Copying file to Runtime directory: " ${FILE}
	)
endforeach()

## C++ compiler options
set(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++14 -Wa,--gstabs -save-temps")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

#add_definitions(-DDEBUG)
## 1. Compile...
add_executable(tim ${SOURCES})
add_dependencies(tim data)

## 2. Link...
target_link_libraries(tim ${OpenCV_LIBS})
