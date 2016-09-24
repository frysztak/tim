cmake_minimum_required(VERSION 3.1)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" "${CMAKE_MODULE_PATH}")
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(OpenCV REQUIRED core imgproc video videoio highgui)
find_package(PkgConfig REQUIRED)
pkg_search_module(nanomsg REQUIRED nanomsg)
find_package(Threads REQUIRED)

## System architecture detection (from x265)
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" SYSPROC)
set(X86_ALIASES x86 i386 i686 x86_64 amd64)
set(ARM_ALIASES armv6l armv7l)
list(FIND X86_ALIASES "${SYSPROC}" X86MATCH)
list(FIND ARM_ALIASES "${SYSPROC}" ARMMATCH)
if("${SYSPROC}" STREQUAL "" OR X86MATCH GREATER "-1")
    set(X86 1)
    if("${CMAKE_SIZEOF_VOID_P}" MATCHES 8)
        set(X64 1)
        add_definitions(-DX86_64=1)
        message(STATUS "Detected x86_64 target processor")
    else()
        message(STATUS "Detected x86 target processor")
    endif()
elseif(ARMMATCH GREATER "-1")
    if(CROSS_COMPILE_ARM)
        message(STATUS "Cross compiling for ARM arch")
    else()
        set(CROSS_COMPILE_ARM 0)
    endif()
    message(STATUS "Detected ARM target processor")
    set(ARM 1)
else()
    message(STATUS "CMAKE_SYSTEM_PROCESSOR value `${CMAKE_SYSTEM_PROCESSOR}` is unknown")
    message(STATUS "Please add this value near ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE}")
endif()

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

option(ASM "Use hand-written assembly instead of intrinsics (bgs only)." OFF)

## Set data dir for json and video files
set(CMAKE_DATA_DIR "${CMAKE_SOURCE_DIR}/data/")
add_definitions(-DDATA_DIR="${CMAKE_DATA_DIR}")

## Find all source files to be compiled into object files
file(GLOB SOURCES src/*.cpp)
if (ASM)
    find_package(Yasm)
    enable_language(ASM_YASM)
    file(GLOB ASM_SOURCES src/sse2/*.asm)
    foreach(ASM ${ASM_SOURCES})
        get_filename_component(fileName "${ASM}" NAME)
        set(outFile ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/tim.dir/src/sse2/${fileName}.o)
        list(APPEND ASM_OBJS "${outFile}")
        add_custom_command(
            OUTPUT ${outFile}
            COMMAND ${YASM_EXECUTABLE} ARGS ${YASM_FLAGS} ${ASM} -o ${outFile} 
            DEPENDS ${ASM})
    endforeach()
else()
    file(GLOB C_SOURCES src/sse2/*.c)
    list(APPEND SOURCES ${C_SOURCES})
endif()

## C++ compiler options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -std=c++14")

## Compile
add_executable(tim ${SOURCES} ${ASM_OBJS})

## Link
target_link_libraries(tim ${OpenCV_LIBS} ${nanomsg_LIBRARIES} Threads::Threads)
