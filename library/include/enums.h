#ifndef __ENUMS_H
#define __ENUMS_H

// ********************************************************************************************************************
// Macro to generate enum values and corresponding strings from a list of (KEY, STRING) pairs
#define GENERATE_ENUM(KEY, STRING) KEY,
#define GENERATE_STRING(KEY, STRING) #STRING,
#define COUNT_ENUM(KEY, STRING) +1


#define DEFINE_ENUM(ENUM_NAME, ENUM_MAP) \
    enum ENUM_NAME { \
        ENUM_MAP(GENERATE_ENUM) \
    }; \
    typedef enum ENUM_NAME ENUM_NAME; \

#define DEFINE_ENUM_FUNCS(ENUM_NAME, ENUM_MAP) \
    int get_##ENUM_NAME##_num() { \
        return 0 ENUM_MAP(COUNT_ENUM); \
    } \
    char *get_##ENUM_NAME##_str(int n) { \
        char *strs[] = { \
            ENUM_MAP(GENERATE_STRING) \
        }; \
        return strs[n]; \
    }
// ********************************************************************************************************************

// Definitions of the enum types and their corresponding string representations
#define GRID_TYPE_MAP(X) \
    X(GRID_TYPE_LCG, LCG) \
    X(GRID_TYPE_FFT, FFT) \
    X(GRID_TYPE_MGRID, MULTIGRID) \
    X(GRID_TYPE_MAZE_LCG, MAZE-LCG) \
    X(GRID_TYPE_MAZE_MGRID, MAZE-MULTIGRID)


#define PARTICLE_POTENTIAL_TYPE_MAP(X) \
    X(PARTICLE_POTENTIAL_TYPE_TF, TF) \
    X(PARTICLE_POTENTIAL_TYPE_LJ, LJ) \
    X(PARTICLE_POTENTIAL_TYPE_SC, SC)


#define CHARGE_ASSIGN_SCHEME_TYPE_MAP(X) \
    X(CHARGE_ASS_SCHEME_TYPE_CIC, CIC) \
    X(CHARGE_ASS_SCHEME_TYPE_SPLQUAD, SPL_QUADR) \
    X(CHARGE_ASS_SCHEME_TYPE_SPLCUB, SPL_CUBIC)


#define INTEGRATOR_TYPE_MAP(X) \
    X(INTEGRATOR_TYPE_OVRVO, OVRVO) \
    X(INTEGRATOR_TYPE_VERLET, VERLET)


#define PRECOND_TYPE_MAP(X) \
    X(PRECOND_TYPE_NONE, NONE) \
    X(PRECOND_TYPE_JACOBI, JACOBI) \
    X(PRECOND_TYPE_MG, MG) \
    X(PRECOND_TYPE_SSOR, SSOR) \
    X(PRECOND_TYPE_BLOCKJACOBI, BLOCKJACOBI)


#define WATER_ELECTROSTATIC_CORR_TYPE_MAP(X) \
    X(WATER_ELECTROSTATIC_CORR_TYPE_SPREAD, SPREAD) \
    X(WATER_ELECTROSTATIC_CORR_TYPE_SR, SR)


#define SMOOTHING_TYPE_MAP(X) \
    X(SMOOTHING_TYPE_NONE, NONE) \
    X(SMOOTHING_TYPE_GAUSS, GAUSS) \
    X(SMOOTHING_TYPE_DIFFUSION, DIFFUSION)


#define PARTICLE_NEIGHBOR_TYPE_MAP(X) \
    X(PARTICLE_NEIGHBOR_TYPE_SPHERE, SPHERE) \
    X(PARTICLE_NEIGHBOR_TYPE_CELL_LIST, CELL_LIST)


// Use macros to generate enums + typedefs
DEFINE_ENUM(grid_type, GRID_TYPE_MAP)
DEFINE_ENUM(potential_type, PARTICLE_POTENTIAL_TYPE_MAP)
DEFINE_ENUM(ca_scheme_type, CHARGE_ASSIGN_SCHEME_TYPE_MAP)
DEFINE_ENUM(integrator_type, INTEGRATOR_TYPE_MAP)
DEFINE_ENUM(precond_type, PRECOND_TYPE_MAP)
DEFINE_ENUM(water_electrostatic_type, WATER_ELECTROSTATIC_CORR_TYPE_MAP)
DEFINE_ENUM(smoothing_type, SMOOTHING_TYPE_MAP)
DEFINE_ENUM(particle_neighbor_type, PARTICLE_NEIGHBOR_TYPE_MAP)


#endif // __ENUMS_H