#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "verlet.h"
#include "constants.h"
#include "mpi_base.h"
#include "linalg.h"
#include "mp_structs.h"

#define EPS_FIELD_TOL 1e-1
#define EPS_FIELD_MAX_ITER 500

#ifdef __cplusplus
#define EXTERN_C extern "C"                                                           
#else
#define EXTERN_C
#endif

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

    long int i;
    long int n2 = size2 * size2;
    long int n3 = size1 * n2;

    double app;
    double *tmp = (double*)malloc(n3 * sizeof(double));
    
    // Compute provisional update for the field phi
    #pragma omp parallel for private(app)
    for (i = 0; i < n3; i++) {
        app = phi[i];
        phi[i] = 2 * app - phi_prev[i];
        phi_prev[i] = app;
    }

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
    int size1, int size2
) {
    int res = -1;
    int iter_conv = 0;

    long int i;
    long int n3 = size1 * size2 * size2;

    double app, constant;
    double *tmp = (double*)malloc(n3 * sizeof(double));
    double *tmp2 = (double*)malloc(n3 * sizeof(double));

    // Compute provisional update for the field phi
    #pragma omp parallel for private(app)
    for (i = 0; i < n3; i++) {
        app = phi[i];
        phi[i] = 2 * app - phi_prev[i];
        phi_prev[i] = app;
    }

    constant = (4 * M_PI) / h;
    laplace_filter(phi, tmp2, size1, size2);
    daxpy(q, tmp2, constant, n3);  // sigma_p = A . phi + 4 * pi * rho / eps
    // memset(y, 0, n3 * sizeof(double));
    // printf("\nprima y = %e\n", norm_inf(y, n3));
    
    // Questo pezzo non e' usato se vedi app e tmp2 vengono riscritti prima di essere letti
    // Serve solo se abiliti il printf sotto
    // laplace_filter(y, tmp, size1, size2);
    // daxpy(tmp2, tmp, -1., n3);  // res = A . y - sigma_p
    // app = norm_inf(tmp, n3);   // Compute norm_inf of residual
    // printf("\ny = %e \t iter=%d \t res=%e\n", norm_inf(y, n3), iter_conv,app);
    
    while(iter_conv < MG_ITER_LIMIT) { 
        // Compute the constraint with the provisional value of the field phi
        multigrid_apply(tmp2, y, size1, size2, get_n_start(), MG_SOLVE_SM); //solve A . y = sigma_p

        laplace_filter(y, tmp, size1, size2);
        daxpy(tmp2, tmp, -1., n3);  // res = A . y - sigma_p
        app = norm_inf(tmp, n3);   // Compute norm_inf of residual
        iter_conv++;
        
        // printf("\ny = %e \t iter=%d \t res=%e\n", norm_inf(y, n3), iter_conv,app);
        
        if (app <= tol){
            res = iter_conv;
            break;
        }
    }
    // Scale the field with the constrained 'force' term
    daxpy(y, phi, -1.0, n3);  // phi = phi - y

    // Free temporary arrays
    free(tmp);
    free(tmp2);

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

    long int i;
    long int n2 = size2 * size2;
    long int n3 = size1 * n2;

    double app;
    double *tmp = (double*)malloc(n3 * sizeof(double));
    
    // Compute provisional update for the field phi
    #pragma omp parallel for private(app)
    for (i = 0; i < n3; i++) {
        app = phi[i];
        phi[i] = 2 * app - phi_prev[i];
        phi_prev[i] = app;
    }

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
@return the number of iterations for convergence of the MG
*/
EXTERN_C int verlet_pb_multigrid(
    double tol, double h, double* phi, double* phi_prev, double* q, double* y,
    int size1, int size2, double *eps_x, double *eps_y, double *eps_z, double *k2_screen
) {
    int res = -1;
    int iter_conv = 0;

    long int i;
    long int n3 = size1 * size2 * size2;

    double app, constant;
    double *tmp = (double*)malloc(n3 * sizeof(double));
    double *tmp2 = (double*)malloc(n3 * sizeof(double));

    // Compute provisional update for the field phi
    #pragma omp parallel for private(app)
    for (i = 0; i < n3; i++) {
        app = phi[i];
        phi[i] = 2 * app - phi_prev[i];
        phi_prev[i] = app;
    }

    constant = (4 * M_PI) / h;
    laplace_filter_pb(phi, tmp2, size1, size2, eps_x, eps_y, eps_z, k2_screen);
    daxpy(q, tmp2, constant, n3);  // sigma_p = A_pb . phi + 4 * pi * q / h
    
    // uncomment below only to print the residual at iteration = 0
    // laplace_filter_pb(y, tmp, size1, size2, eps_x, eps_y, eps_z, k2_screen);
    // daxpy(tmp2, tmp, -1., n3);  // res = A_pb . y - sigma_p
    // app = norm_inf(tmp, n3);   // Compute norm_inf of residual
    // printf("\ny = %e \t iter=%d \t res=%e\n", norm_inf(y, n3), iter_conv,app);
    
    while(iter_conv < MG_ITER_LIMIT_PB) { 
        // Compute the constraint with the provisional value of the field phi
        multigrid_pb_apply(tmp2, y, size1, size2, get_n_start(), MG_SOLVE_SM_PB, eps_x, eps_y, eps_z, k2_screen); //solve A_pb . y = sigma_p

        laplace_filter_pb(y, tmp, size1, size2, eps_x, eps_y, eps_z, k2_screen);
        daxpy(tmp2, tmp, -1., n3);  // res = A . y - sigma_p
        app = norm_inf(tmp, n3);   // Compute norm_inf of residual
        iter_conv++;
        
        // printf("\ny = %e \t iter=%d \t res=%e\n", norm_inf(y, n3), iter_conv,app);
        
        if (app <= tol){
            res = iter_conv;
            break;
        }
    }
    // Scale the field with the constrained 'force' term
    daxpy(y, phi, -1.0, n3);  // phi = phi - y

    // Free temporary arrays
    free(tmp);
    free(tmp2);

    if (res == -1) {
        fprintf(stderr, "Warning: Multigrid did not converge after 1000 iterations.\n");    
    }

    return res;
}

EXTERN_C int verlet_pb_multigrid_eps_field(
    double tol, double h, double* phi, double* phi_prev, double* q, double* y,
    int size1, int size2, double *eps_x, double *eps_y, double *eps_z, double *k2_screen,
    grid *grid_ctx
) {
    if (grid_ctx == NULL || grid_ctx->kBT <= 0.0) {
        mpi_printf("Warning: field-dependent dielectric requested without valid kBT. Running single PB solve.\n");
    }
    int res = -1;
    int iter_conv = 0;

    long int i;
    long int n3 = (long int)size1 * size2 * size2;

    double app, constant;
    double *tmp = (double*)malloc(n3 * sizeof(double));
    double *tmp2 = (double*)malloc(n3 * sizeof(double));

    // Provisional Verlet update
    #pragma omp parallel for private(app)
    for (i = 0; i < n3; i++) {
        app = phi[i];
        phi[i] = 2 * app - phi_prev[i];
        phi_prev[i] = app;
    }

    constant = (4 * M_PI) / h;

    double max_diff = EPS_FIELD_TOL + 1.0;
    int eps_iter = 0;

    while (max_diff > EPS_FIELD_TOL && eps_iter < EPS_FIELD_MAX_ITER) {
        if (grid_ctx != NULL && grid_ctx->kBT > 0.0) {
            max_diff = grid_update_eps_field_dependent(grid_ctx, NULL, grid_ctx->kBT); 
        } else {
            max_diff = 0.0;  // Skip epsilon loop if no context
        }

        // Recompute sigma_p = A_pb . phi + 4*pi*q/h with updated eps
        laplace_filter_pb(phi, tmp2, size1, size2, eps_x, eps_y, eps_z, k2_screen);
        daxpy(q, tmp2, constant, n3);

        res = -1;
        iter_conv = 0;
        while(iter_conv < MG_ITER_LIMIT_PB) { 
            multigrid_pb_apply(tmp2, y, size1, size2, get_n_start(), MG_SOLVE_SM_PB, eps_x, eps_y, eps_z, k2_screen); //solve A_pb . y = sigma_p

            laplace_filter_pb(y, tmp, size1, size2, eps_x, eps_y, eps_z, k2_screen);
            daxpy(tmp2, tmp, -1., n3);  // res = A . y - sigma_p
            app = norm_inf(tmp, n3);   // Compute norm_inf of residual
            iter_conv++;
            
            // printf("\ny = %e \t iter=%d \t res=%e\n", norm_inf(y, n3), iter_conv,app);
            
            if (app <= tol){
                res = iter_conv;
                break;
            }
        }
        daxpy(y, phi, -1.0, n3);  // phi = phi - y

        eps_iter++;
        mpi_printf("Field-dependent dielectric iteration %d: max diff = %e\n", eps_iter, max_diff);

        /* Stop early if we were not actually iterating eps */
        if (grid_ctx == NULL || grid_ctx->kBT <= 0.0) {
            break;
        }
    }

    free(tmp);
    free(tmp2);

    if (max_diff > EPS_FIELD_TOL && grid_ctx != NULL && grid_ctx->kBT > 0.0) {
        mpi_printf("Warning: dielectric update did not converge after %d iterations (max diff = %e)\n", eps_iter, max_diff);
    } else if (grid_ctx != NULL && grid_ctx->kBT > 0.0) {
        mpi_printf(
            "Field-dependent dielectric converged in %d iterations (max diff = %e)\n",
            eps_iter, max_diff
        );
    }

    if (res == -1) {
        fprintf(stderr, "Warning: Multigrid did not converge after %d iterations.\n", MG_ITER_LIMIT_PB);
    }

    return res;
}
