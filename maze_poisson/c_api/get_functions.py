"""Collection of library functions to extract data from the C API"""
import ctypes

import numpy as np
import numpy.ctypeslib as npct

from . import capi

# int get_grid_type_num() {
capi.register_function(
    'get_grid_type_num', ctypes.c_int, [],
)

# char *get_grid_type_str(int n) {
capi.register_function(
    'get_grid_type_str', ctypes.c_char_p, [
        ctypes.c_int,
    ],
)

# int get_potential_type_num() {
capi.register_function(
    'get_potential_type_num', ctypes.c_int, [],
)

# char *get_potential_type_str(int n) {
capi.register_function(
    'get_potential_type_str', ctypes.c_char_p, [
        ctypes.c_int,
    ],
)

# int get_ca_scheme_type_num() {
capi.register_function(
    'get_ca_scheme_type_num', ctypes.c_int, [],
)

# char *get_ca_scheme_type_str(int n) {
capi.register_function(
    'get_ca_scheme_type_str', ctypes.c_char_p, [
        ctypes.c_int,
    ],
)

# int get_integrator_type_num() {
capi.register_function(
    'get_integrator_type_num', ctypes.c_int, [],
)

# char *get_integrator_type_str(int n) {
capi.register_function(
    'get_integrator_type_str', ctypes.c_char_p, [
        ctypes.c_int,
    ],
)

# int get_precond_type_num() {
capi.register_function(
    'get_precond_type_num', ctypes.c_int, [],
)

# char *get_precond_type_str(int n) {
capi.register_function(
    'get_precond_type_str', ctypes.c_char_p, [
        ctypes.c_int,
    ],
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

# void get_fcs_db(double *recv) {
capi.register_function(
    'get_fcs_db', None, [
        npct.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
    ],
)

# void get_fcs_ib(double *recv) {
capi.register_function(
    'get_fcs_ib', None, [
        npct.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
    ],
)

# void get_fcs_np(double *recv) {
capi.register_function(
    'get_fcs_np', None, [
        npct.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
    ],
)

# void get_fcs_tot(double *recv) {
capi.register_function(
    'get_fcs_tot', None, [
        npct.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
    ],
)

# void get_charges(double *recv) {
capi.register_function(
    'get_charges', None, [
        npct.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    ],
)

# void get_masses(double *recv) {
capi.register_function(
    'get_masses', None, [
        npct.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    ],
)

# void get_radii(double *recv) {
capi.register_function(
    'get_radii', None, [
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

# double get_energy_elec() {
capi.register_function(
    'get_energy_elec', ctypes.c_double, [],
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
