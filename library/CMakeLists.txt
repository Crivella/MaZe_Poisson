cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(maze_poisson
    VERSION 0.1.0
    LANGUAGES C
)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

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

# Find the required packages
find_package(OpenMP REQUIRED C)
find_package(FFTW3 REQUIRED COMPONENTS DOUBLE_OPENMP DOUBLE)

# Set the compiler flags
set(CMAKE_C_FLAGS_DEBUG "-g")
set(CMAKE_C_FLAGS_RELEASE "-O3")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O3 -g")


if(MP_FAST_MATH)
    if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --fast-math")
    elseif(CMAKE_C_COMPILER_ID STREQUAL "Clang")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math")
    else()
        message(WARNING "Fast math not supported for this compiler")
    endif()
endif()

if(MP_MTUNE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mtune=native")
endif()

if (MP_AVX)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2")
endif()

# add_subdirectory(src)

# Create the shared library
add_library(maze_poisson SHARED
    src/linalg.c
    src/laplace.c
    src/forces.c
)

target_include_directories(maze_poisson PRIVATE include)

# Set OpenMP flags 
if (OpenMP_C_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif()
# Set FFTW flags
if (FFTW3_FOUND)
    target_include_directories(maze_poisson PRIVATE ${FFTW_INCLUDES})
    target_link_libraries(maze_poisson PRIVATE ${FFTW3_LIBRARIES})
    target_sources(maze_poisson PRIVATE src/fftw_wrap.c)
endif()

install(TARGETS maze_poisson
    LIBRARY DESTINATION .
)
