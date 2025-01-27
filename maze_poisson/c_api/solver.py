import ctypes

import numpy as np
import numpy.ctypeslib as npct

from . import capi

capi.register_function(
    'solver_initialize', None, [
        ctypes.c_int,
    ],
)

# void solverinitialize_grid(int n_grid, double L, double h, double tol, int grid_type) {
capi.register_function(
    'solver_initialize_grid', None, [
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
    ],
)

# void solverinitialize_particles(
#     int n, double L, double h, int n_p, int pot_type,
#     double *pos, double *vel, double *mass, long int *charges
# ) {
capi.register_function(
    'solver_initialize_particles', None, [
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        npct.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.int64, ndim=1, flags='C_CONTIGUOUS'),
    ],
)

# void solverinitialize_integrator(int n_p, double dt, double T, double gamma, int itg_type, int itg_enabled) {
capi.register_function(
    'solver_initialize_integrator', None, [
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
    ],
)

# int solver_update_charges() {
capi.register_function(
    'solver_update_charges', ctypes.c_int, [],
)

# void solver_init_field() {
capi.register_function(
    'solver_init_field', None, [],
)

# int solver_update_field() {
capi.register_function(
    'solver_update_field', ctypes.c_int, [],
)

# void solver_compute_forces_elec() {
capi.register_function(
    'solver_compute_forces_elec', None, [],
)

# double solver_compute_forces_noel() {
capi.register_function(
    'solver_compute_forces_noel', ctypes.c_double, [],
)

# void solver_compute_forces_tot() {
capi.register_function(
    'solver_compute_forces_tot', None, [],
)

# void integrator_part_1() {
capi.register_function(
    'integrator_part_1', None, [],
)

# void integrator_part_2() {
capi.register_function(
    'integrator_part_2', None, [],
)

# # int solver_nitialize_md(int preconditioning, int vel_rescale) {
# capi.register_function(
#     'solver_initialize_md', ctypes.c_int, [
#         ctypes.c_int,
#         ctypes.c_int,
#     ],
# )

# # void solver_md_loop_iter() {
# capi.register_function(
#     'solver_md_loop_iter', None, [],
# )

# int solver_check_thermostat() {
capi.register_function(
    'solver_check_thermostat', ctypes.c_int, [],
)

# void solver_rescale_velocities() {
capi.register_function(
    'solver_rescale_velocities', None, [],
)

# # void solver_run_n_steps(int n_steps) {
# capi.register_function(
#     'solver_run_n_steps', None, [
#         ctypes.c_int,
#     ],
# )

# void solver_finalize() {
capi.register_function(
    'solver_finalize', None, [],
)

# void get_pos(double *recv) {
capi.register_function(
    'get_pos', None, [
        npct.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
    ],
)

# void get_vel(double *recv) {
capi.register_function(
    'get_vel', None, [
        npct.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
    ],
)

# void get_fcs_elec(double *recv) {
capi.register_function(
    'get_fcs_elec', None, [
        npct.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
    ],
)

# void get_fcs_noel(double *recv) {
capi.register_function(
    'get_fcs_noel', None, [
        npct.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
    ],
)

# void get_fcs_tot(double *recv) {
capi.register_function(
    'get_fcs_tot', None, [
        npct.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
    ],
)

# void get_charges(long int *recv) {
capi.register_function(
    'get_charges', None, [
        npct.ndpointer(dtype=np.int64, ndim=1, flags='C_CONTIGUOUS'),
    ],
)

# void get_masses(double *recv) {
capi.register_function(
    'get_masses', None, [
        npct.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    ],
)

# void get_field(double *recv) {
capi.register_function(
    'get_field', None, [
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    ],
)

# void get_field_prev(double *recv) {
capi.register_function(
    'get_field_prev', None, [
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    ],
)

# void solver_set_field(double *phi) {
capi.register_function(
    'solver_set_field', None, [
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    ],
)

# void solver_set_field_prev(double *phi) {
capi.register_function(
    'solver_set_field_prev', None, [
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    ],
)

# void get_q(double *recv) {
capi.register_function(
    'get_q', None, [
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    ],
)

# double get_kinetic_energy() {
capi.register_function(
    'get_kinetic_energy', ctypes.c_double, [],
)

# void get_momentum(double *recv) {
capi.register_function(
    'get_momentum', None, [
        npct.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    ],
)

# double get_temperature() {
capi.register_function(
    'get_temperature', ctypes.c_double, [],
)
