cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(maze_poisson
    VERSION 0.1.0
    LANGUAGES C
)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Require FFTW3
find_package(FFTW3 REQUIRED COMPONENTS DOUBLE_OPENMP DOUBLE)

# Set the build type to Release
set(CMAKE_BUILD_TYPE Release)

# Set the output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Options
option(MP_FAST_MATH "Enable fast math" ON)
option(MP_USE_OPENMP "Use OpenMP" ON)
option(MP_MTUNE "Enable -mtune=native" ON)
option(MP_AVX "Enable AVX" ON)

# Set the compiler flags
set(CMAKE_C_FLAGS "-Wall -Wextra -Wpedantic -Werror")
set(CMAKE_C_FLAGS_DEBUG "-g")
set(CMAKE_C_FLAGS_RELEASE "-O3")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O3 -g")

# Set OpenMP flags 

if(MP_FAST_MATH)
    # Check for GCC vs LLVM/Clang
    if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --fast-math")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math")
    endif()
endif()

if(MP_USE_OPENMP)
    find_package(OpenMP REQUIRED C)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif()

if(MP_MTUNE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mtune=native")
endif()

if (MP_AVX)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx")
endif()

# Include the FFTW3 header files
include_directories(${FFTW_INCLUDES})

# Create the shared library
add_library(maze_poisson SHARED
    laplace.c
    fftw_wrap.c
)

# TODO: Need to generalize instead of hard code and consider also combination of FFTW3 and FFTW3_OMP
target_link_libraries(maze_poisson 
    # INTERFACE
    # FFTW3
    fftw3_omp
    fftw3
    m
)

install(TARGETS maze_poisson
    LIBRARY DESTINATION .
)
