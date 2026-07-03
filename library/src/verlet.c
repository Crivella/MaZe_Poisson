#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "verlet.h"
#include "constants.h"
#include "mp_structs.h"
#include "mpi_base.h"
#include "linalg.h"

#ifdef __cplusplus
#define EXTERN_C extern "C"                                                           
#else
#define EXTERN_C
#endif

/*
Compute the provisional update for the field phi using the Verlet algorithm.
- phi(t) = 2 * phi(t-1) - phi(t-2)
@param phi: the potential field at the current time step
@param phi_prev: the potential field at the previous time step
@param size: the total number of grid points
*/
void verlet_update(double *phi, double *phi_prev, long int size) {
    double app;
    #pragma omp parallel for private(app)
    for (long int i = 0; i < size; i++) {
        app = phi[i];
        phi[i] = 2 * app - phi_prev[i];
        phi_prev[i] = app;
    }
}

// Warm-start coefficients: base, Verlet-like linear, and second-order extrapolation.
static const double POLY_COEF[MAZE_Y_HIST_MAX + 1][MAZE_Y_HIST_MAX + 1] = {
    {1.0},
    {2.0, -1.0},
    {3.0, -3.0, 1.0},
};

static void fill_extrapolation_coefficients(const y_extrap_config *y_extrap, int y_hist_len, double *coef, int *order) {
    for (int i = 0; i <= MAZE_Y_HIST_MAX; i++) {
        coef[i] = 0.0;
    }

    int effective_order = y_extrap->order;
    if (effective_order > y_hist_len) {
        effective_order = y_hist_len;
    }

    for (int i = 0; i <= effective_order; i++) {
        coef[i] = POLY_COEF[effective_order][i];
    }
    *order = effective_order;
}

// Evaluate an order-p predictor using y^{k-1} and older history terms.
static void extrap_predict(double *out, double *y_km1, double **y_hist, const double *c, int p, long int size) {
    vec_copy(y_km1, out, size);
    dscal(out, c[0], size);
    for (int j = 1; j <= p; j++) {
        daxpy(y_hist[j - 1], out, c[j], size);
    }
}

static void rotate_y_history(double **y_hist) {
    double *new_hist[MAZE_Y_HIST_MAX + 1];
    new_hist[0] = y_hist[MAZE_Y_HIST_MAX];
    for (int i = 1; i < MAZE_Y_HIST_MAX; i++) {
        new_hist[i] = y_hist[i - 1];
    }
    new_hist[MAZE_Y_HIST_MAX] = y_hist[MAZE_Y_HIST_MAX - 1];
    for (int i = 0; i <= MAZE_Y_HIST_MAX; i++) {
        y_hist[i] = new_hist[i];
    }
}

// Snapshot y^{k-1} and overwrite y with the selected warm-start predictor.
static void y_build_guess(double *y, double **y_hist, const y_extrap_config *y_extrap, int y_hist_len, long int size) {
    double *y_km1 = y_hist[MAZE_Y_HIST_MAX];  // spare slot -> snapshot of y^{k-1}
    double coef[MAZE_Y_HIST_MAX + 1];
    int order;
    vec_copy(y, y_km1, size);
    fill_extrapolation_coefficients(y_extrap, y_hist_len, coef, &order);
    extrap_predict(y, y_km1, y_hist, coef, order, size);
}

static void y_shift_history(double **y_hist, int *y_hist_len) {
    rotate_y_history(y_hist);
    if (*y_hist_len < MAZE_Y_HIST_MAX) {
        (*y_hist_len)++;
    }
}

/*
Apply Verlet algorithm to compute the updated value of the field phi, with LCG + SHAKE.
The previous and current fields and the y array are updated in place.
@param tol: tolerance
@param h: the grid spacing
@param phi: the potential field of size n_grid * n_grid * n_grid
@param phi_prev: electrostatic field for step t - 1 Verlet
@param q: the charge on a grid of size n_grid * n_grid * n_grid\
@param y: copy of the 'q' given as input to the function
@param n_grid: the number of grid points in each dimension
@param precond: the preconditioner function

@return the number of iterations for convergence of the LCG
*/
EXTERN_C int verlet_poisson(
    double tol, double h, double* phi, double* phi_prev, double* q, double* y,
    int size1, int size2,
    void (*precond)(double *, double *, int, int, int)
) {
    int iter_conv;
    long int n3 = size1 * size2 * size2;

    double *tmp = (double*)malloc(n3 * sizeof(double));
    
    // Compute provisional update for the field phi
    verlet_update(phi, phi_prev, n3);

    // Compute the constraint with the provisional value of the field phi
    laplace_filter(phi, tmp, size1, size2);
    daxpy(q, tmp, (4 * M_PI) / h, n3);  // sigma_p = A . phi + 4 * pi * rho / eps

    // Apply LCG
    if (precond == NULL) {
        iter_conv = conj_grad(tmp, y, y, tol, size1, size2);  // Inplace y <- y0 - tolerance scaled by 4*pi/h to guarantee correct sigma_p = A . phi . h/4pi + rho/eps 
    } else {
        iter_conv = conj_grad_precond(tmp, y, y, tol, size1, size2, precond);  // Inplace y <- y0 - tolerance scaled by 4*pi/h to guarantee correct sigma_p = A . phi . h/4pi + rho/eps 
    }

    // Scale the field with the constrained 'force' term
    daxpy(y, phi, -1.0, n3);  // phi = phi - y

    // Free temporary arrays
    free(tmp);

    return iter_conv;
}

/*
Apply Verlet algorithm to compute the updated value of the field phi, with Multigrid + SHAKE.
The previous and current fields and the y array are updated in place.
@param tol: tolerance
@param h: the grid spacing
@param phi: the potential field of size n_grid * n_grid * n_grid
@param phi_prev: electrostatic field for step t - 1 Verlet
@param q: the charge on a grid of size n_grid * n_grid * n_grid\
@param y: copy of the 'q' given as input to the function
@param n_grid: the number of grid points in each dimension

@return the number of iterations for convergence of the LCG
*/
EXTERN_C int verlet_poisson_multigrid(
    double tol, double h, double* phi, double* phi_prev, double* q, double* y,
    double** y_hist, const y_extrap_config *y_extrap,
    int *y_hist_len,
    int size1, int size2
) {
    int res = -1;

    long int n3 = size1 * size2 * size2;

    double constant;
    double *tmp = (double*)malloc(n3 * sizeof(double));

    // Compute provisional update for the field phi
    verlet_update(phi, phi_prev, n3);

    constant = (4 * M_PI) / h;
    laplace_filter(phi, tmp, size1, size2);
    daxpy(q, tmp, constant, n3);  // sigma_p = A . phi + 4 * pi * rho / eps
    // memset(y, 0, n3 * sizeof(double));
    // printf("\nprima y = %e\n", norm_inf(y, n3));

    // Build the y_0 initial guess for the multigrid solve.
    y_build_guess(y, y_hist, y_extrap, *y_hist_len, n3);

    res = multigrid_solve(tol, tmp, y, size1, size2, get_n_start());

    // Keep y^{k-1} available for the next warm-start predictor.
    y_shift_history(y_hist, y_hist_len);

    // Scale the field with the constrained 'force' term
    daxpy(y, phi, -1.0, n3);  // phi = phi - y

    // Free temporary arrays
    free(tmp);

    if (res == -1) {
        fprintf(stderr, "Warning: Multigrid did not converge after 1000 iterations.\n");    
    }

    return res;
}


EXTERN_C int verlet_poisson_pb(
    double tol, double h, double* phi, double* phi_prev, double* q, double* y,
    int size1, int size2,
    double *eps_x, double *eps_y, double *eps_z, double *k2_screen
) {
    int iter_conv;
    long int n3 = size1 * size2 * size2;

    double *tmp = (double*)malloc(n3 * sizeof(double));
    
    // Compute provisional update for the field phi
    verlet_update(phi, phi_prev, n3);

    // Compute the constraint with the provisional value of the field phi
    laplace_filter_pb(
        phi, tmp, size1, size2,
        eps_x, eps_y, eps_z, k2_screen
    );
    daxpy(q, tmp, (4 * M_PI) / h, n3);  // sigma_p = A . phi + 4 * pi * rho

    // Apply LCG
    iter_conv = conj_grad_pb(
        tmp, y, y, tol, size1, size2,
        eps_x, eps_y, eps_z, k2_screen
    );  // Inplace y <- y0

    // Scale the field with the constrained 'force' term
    daxpy(y, phi, -1.0, n3);  // phi = phi - y

    // Free temporary arrays
    free(tmp);

    return iter_conv;
}


/*
Apply Verlet algorithm to compute the updated value of the field phi, with Multigrid + SHAKE.
The previous and current fields and the y array are updated in place.
@param tol: tolerance
@param h: the grid spacing
@param phi: the potential field of size n_grid * n_grid * n_grid
@param phi_prev: electrostatic field for step t - 1 Verlet
@param q: the charge on a grid of size n_grid * n_grid * n_grid\
@param y: copy of the 'q' given as input to the function
@param n_grid: the number of grid points in each dimension
@param eps_x, eps_y, eps_z: the spatially dependent dielectric constants in each direction
@param k2_screen: the spatially dependent screening term for the linearized PB equation
@return the number of iterations for convergence of the MG or -1 if MG did not converge
*/
EXTERN_C int verlet_pb_multigrid(
    double tol, double h, double* phi, double* phi_prev, double* q, double* y,
    double** y_hist, const y_extrap_config *y_extrap,
    int *y_hist_len,
    int size1, int size2, double *eps_x, double *eps_y, double *eps_z, double *k2_screen
) {
    int res = -1;

    long int n3 = size1 * size2 * size2;

    double constant;
    double *tmp = (double*)malloc(n3 * sizeof(double));

    // Compute provisional update for the field phi
    verlet_update(phi, phi_prev, n3);

    constant = (4 * M_PI) / h;
    laplace_filter_pb(phi, tmp, size1, size2, eps_x, eps_y, eps_z, k2_screen);
    daxpy(q, tmp, constant, n3);  // sigma_p = A_pb . phi + 4 * pi * q / h = sigma_p^k

    // Build the y_0 initial guess for the multigrid solve.
    y_build_guess(y, y_hist, y_extrap, *y_hist_len, n3);

    multigrid_solve_pb(
        tol, tmp, y, size1, size2, get_n_start(),
        eps_x, eps_y, eps_z, k2_screen
    );  // solve A_pb . y = sigma_p

    // Keep y^{k-1} available for the next warm-start predictor.
    y_shift_history(y_hist, y_hist_len);

    // Scale the field with the constrained 'force' term
    daxpy(y, phi, -1.0, n3);  // phi = phi - y

    // Free temporary arrays
    free(tmp);

    if (res == -1) {
        fprintf(stderr, "Warning: Multigrid did not converge after 1000 iterations.\n");    
    }

    return res;
}
