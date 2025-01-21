#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

/*
# apply Verlet algorithm to compute the updated value of the field phi, with LCG + SHAKE
@param tol: tolerance
@param h: the grid spacing
@param phi: the potential field of size n_grid * n_grid * n_grid
@param phi_prev: electrostatic field for step t - 1 Verlet
@param q: the charge on a grid of size n_grid * n_grid * n_grid\
@param debug: flag for debugging
@param y: copy of the 'q' given as input to the function
@param n_grid: the number of grid points in each dimension

@return updated value of phi
*/
void VerletPoisson(double tol, double h, double* phi, double* phi_prev, double* q, bool debug, double* y, int n_grid, double* y_new, int* iter_conv) {
    double tol = tol;
    double h = h;
    int size = n_grid * n_grid * n_grid;
    double* phi_temp = (double*)malloc(size * sizeof(double));
    double* sigma_p = (double*)malloc(size * sizeof(double));
    double* matrix_mult_result = (double*)malloc(size * sizeof(double));
    
    // Compute provisional update for the field phi
    for (int i = 0; i < size; i++) {
        phi_temp[i] = phi[i];
        phi[i] = 2 * phi[i] - phi_prev[i];
        phi_prev[i] = phi_temp[i];
    }

    // Compute the constraint with the provisional value of the field phi
    MatrixVectorProduct(phi, matrix_mult_result, size);
    for (int i = 0; i < size; i++) {
        sigma_p[i] = q[i] / h + matrix_mult_result[i] / (4 * M_PI);
    }

    // Apply LCG
    *iter_conv = PrecondLinearConjGradPoisson(sigma_p, y_new, size, tol);

    // Scale the field with the constrained 'force' term
    for (int i = 0; i < size; i++) {
        phi[i] -= y_new[i] * (4 * M_PI);
    }

    if (debug) {
        double* debug_matrix_mult_result = (double*)malloc(size * sizeof(double));
        MatrixVectorProduct(y_new, debug_matrix_mult_result, size);
        
        double max_precision = 0.0;
        for (int i = 0; i < size; i++) {
            double diff = fabs(debug_matrix_mult_result[i] - sigma_p[i]);
            if (diff > max_precision) {
                max_precision = diff;
            }
        }
        printf("LCG precision     : %f\n", max_precision);

        MatrixVectorProduct(phi, debug_matrix_mult_result, size);
        double max_constraint = 0.0;
        for (int i = 0; i < size; i++) {
            double constraint = fabs(q[i] / h + debug_matrix_mult_result[i] / (4 * M_PI));
            if (constraint > max_constraint) {
                max_constraint = constraint;
            }
        }
        printf("max of constraint: %f\n", max_constraint);

        free(debug_matrix_mult_result);
    }

    // Free temporary arrays
    free(phi_temp);
    free(sigma_p);
    free(matrix_mult_result);
}
