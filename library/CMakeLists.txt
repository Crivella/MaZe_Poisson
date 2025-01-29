cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(maze_poisson
    VERSION 0.1.0
    LANGUAGES C
)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Set the default build type to Release if not set
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set the output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Options
option(MP_FAST_MATH "Enable fast math" ON)
option(MP_MTUNE "Enable -mtune=native" ON)
option(MP_AVX "Enable AVX" ON)

# Find the required packages
find_package(OpenMP COMPONENTS C)
find_package(MPI COMPONENTS C)
find_package(FFTW3 COMPONENTS DOUBLE_OPENMP DOUBLE)
find_package(LAPACK)

# Set the compiler flags
set(CMAKE_C_FLAGS_DEBUG "-g")
set(CMAKE_C_FLAGS_RELEASE "-O3")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O3 -g")

# Create the shared library
add_library(maze_poisson SHARED
    src/linalg.c
    src/laplace.c
    src/forces.c
    src/charges.c
    src/fftw_wrap.c
    
    src/omp_base.c
    src/mpi_base.c

    src/particles.c

    src/grid_general.c
    src/grid_lcg.c
    src/grid_fft.c
    
    src/integrator_general.c
    src/integrator_ovrvo.c
    src/integrator_verlet.c

    src/solver_md.c
)

if(MP_FAST_MATH)
    if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
        target_compile_options(maze_poisson PRIVATE --fast-math)
    elseif(CMAKE_C_COMPILER_ID STREQUAL "Clang")
        target_compile_options(maze_poisson PRIVATE -ffast-math)
    elseif(CMAKE_C_COMPILER_ID STREQUAL "Intel")
        target_compile_options(maze_poisson PRIVATE -ffast-math)
    else()
        message(WARNING "Fast math not supported for this compiler ${CMAKE_C_COMPILER_ID}")
    endif()
endif()

if(MP_MTUNE)
    target_compile_options(maze_poisson PRIVATE -mtune=native)
endif()

if (MP_AVX)
    target_compile_options(maze_poisson PRIVATE -mavx2)
endif()


target_include_directories(maze_poisson PRIVATE include)

# Set OpenMP flags 
if (OpenMP_C_FOUND)
    target_compile_options(maze_poisson PRIVATE ${OpenMP_C_FLAGS})
    target_link_libraries(maze_poisson PRIVATE ${OpenMP_C_LIBRARIES})
endif()
# Set MPI flags
if (MPI_C_FOUND)
    target_compile_definitions(maze_poisson PRIVATE __MPI)
    target_include_directories(maze_poisson PRIVATE ${MPI_C_INCLUDE_PATH})
    target_link_libraries(maze_poisson PRIVATE ${MPI_C_LIBRARIES})
endif()
# Set LAPACK flags
if (LAPACK_FOUND)
    target_compile_definitions(maze_poisson PRIVATE __LAPACK)
    list(APPEND _lapack_libs
        ${BLAS_LIBRARIES}
        ${BLAS_LINKER_FLAGS}
        ${LAPACK_LIBRARIES}
        ${LAPACK_LINKER_FLAGS})
    list(REMOVE_DUPLICATES _lapack_libs)
    target_link_libraries(maze_poisson PRIVATE ${_lapack_libs})
    find_path(
        LAPACK_INCLUDE_DIRS
        NAMES cblas.h
        HINTS $ENV{LAPACK_ROOT}
        PATH_SUFFIXES include include/flexiblas/
        NO_DEFAULT_PATH    
    )
    message(STATUS "LAPACK include dirs: ${LAPACK_INCLUDE_DIRS}")
    target_include_directories(maze_poisson PRIVATE ${LAPACK_INCLUDE_DIRS})
endif()
# Set FFTW flags
if (FFTW3_FOUND)
    target_compile_definitions(maze_poisson PRIVATE __FFTW)
    target_link_libraries(maze_poisson PRIVATE FFTW3)
endif()

install(TARGETS maze_poisson
    LIBRARY DESTINATION .
)
