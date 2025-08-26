#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "linalg.h"
#include "verlet.h"
#include "mp_structs.h"
#include "mpi_base.h"

#define JACOBI_OMEGA 0.66


int v_cycle(double *in, double *out, int s1, int s2, int n_start, int sm, int depth) {
    int res;
    int s1_nxt, s2_nxt, n_start_nxt;
    long int size = s1 * s2 * s2;

    s1_nxt = (s1 + 1 - (n_start % 2)) / 2;
    s2_nxt = s2 / 2;
    n_start_nxt = (n_start + 1) / 2;

    if (s1_nxt < fmax(4, get_size())) {
        if (depth == 0) {
            mpi_fprintf(stderr, "------------------------------------------------------------------------------------\n");
            mpi_fprintf(stderr, "Multigrid: requires atleast one level of recursion (s1 >= %d)\n", fmax(4, get_size()));
            mpi_fprintf(stderr, "Increase the size of the grid or reduce the number of MPI processes.\n");
            mpi_fprintf(stderr, "------------------------------------------------------------------------------------\n");
            exit(1);
        }
        // Base case: no more levels to go down
        smooth(in, out, s1, s2, sm);  // out = smooth(in, out)  ~solve(A . out = in)
        return depth;
    }

    long int size_nxt = s1_nxt * s2_nxt * s2_nxt;

    double *r = mpi_grid_allocate(s1, s2);
    double *rhs = mpi_grid_allocate(s1_nxt, s2_nxt);
    double *eps = mpi_grid_allocate(s1_nxt, s2_nxt);

    memset(eps, 0, size_nxt * sizeof(double));  // eps = 0

    smooth(in, out, s1, s2, sm);  // out = smooth(in, out)  ~solve(A . out = in)
    // r  =  in - A . out
    laplace_filter(out, r, s1, s2);
    dscal(r, -1.0, size);
    daxpy(in, r, 1.0, size);

    restriction(r, rhs, s1, s2, n_start);  // rhs = restriction(r)
    // smooth(rhs, eps, s1_nxt, s2_nxt, 2 * sm);  // eps = smooth(rhs)  ~solve(A . eps = rhs)
    res = v_cycle(rhs, eps, s1_nxt, s2_nxt, n_start_nxt, 2 * sm, depth + 1);  // eps = v_cycle(rhs)

    prolong(eps, r, s1_nxt, s2_nxt, s1, s2, n_start);  // r = prolong(eps)
    daxpy(r, out, 1.0, size);  // out = out + r

    smooth(in, out, s1, s2, sm);  // out = smooth(in, out)  ~solve(A . out = in)

    mpi_grid_free(r, s2);
    mpi_grid_free(rhs, s2_nxt);
    mpi_grid_free(eps, s2_nxt);

    return res;  // Return the number of levels processed
}

int multigrid_apply_recursive(
    double *in, double *out, int s1, int s2, int n_start1,
    int sm1, int sm2, int sm3, int sm4
) {
    // memset(out, 0, s1 * s2 * s2 * sizeof(double));  // Initialize out to zero
    return v_cycle(in, out, s1, s2, n_start1, sm1, 0);  // Apply the recursive V-cycle multigrid method
}

/*
Apply the multigrid method to solve the Poisson equation  A.out = in
using a 3-level V-cycle multigrid method.

@param in: input array (right-hand side of the equation)
@param out: in/out array (starting guess/solution)
@param s1: size of the first dimension (number of slices)
@param s2: size of the second dimension (number of grid points per slice)
@param n_start1: starting index for the first dimension (used for restriction)
*/
void multigrid_apply_3lvl(
    double *in, double *out, int s1, int s2, int n_start1,
    int sm1, int sm2, int sm3, int sm4
) {
    int n1 = s2;
    int n2 = n1 / 2;
    int n3 = n2 / 2;

    int n_loc1 = s1;

    int n_loc2 = (n_loc1 + 1 - (n_start1 % 2)) / 2;
    int n_start2 = (n_start1 + 1) / 2;

    int n_loc3 = (n_loc2 + 1 - (n_start2 % 2)) / 2;
    // int n_start3 = (n_start2 + 1) / 2;

    if (n_loc3 == 0) {
        mpi_fprintf(stderr, "------------------------------------------------------------------------------------\n");
        mpi_fprintf(stderr, "Warning: after restriction some processors have no local grid points!\n");
        mpi_fprintf(stderr, "This case is not yet implemented, please use MULTIGRID with atleast 4 slices\n");
        mpi_fprintf(stderr, "per processor (N_grid / num_mpi_procs >= 4) \n");
        mpi_fprintf(stderr, "------------------------------------------------------------------------------------\n");
        exit(1);
    }

    long int size1 = n_loc1 * n1 * n1;
    long int size2 = n_loc2 * n2 * n2;
    long int size3 = n_loc3 * n3 * n3;

    double *r1 = mpi_grid_allocate(n_loc1, n1);
    double *r2 = mpi_grid_allocate(n_loc2, n2);
    double *r3 = mpi_grid_allocate(n_loc3, n3);

    double *e2 = mpi_grid_allocate(n_loc2, n2);
    double *e3 = mpi_grid_allocate(n_loc3, n3);

    double *tmp2 = mpi_grid_allocate(n_loc2, n2);

    // memset(out, 0, size1 * sizeof(double));  // out = 0
    memset(e2, 0, size2 * sizeof(double));  // e2 = 0
    memset(e3, 0, size3 * sizeof(double));  // e3 = 0

    smooth(in, out, n_loc1, n1, sm1);  // out = smooth(in, out)  ~solve(A . out = in)
    // r1  =  in - A . out
    laplace_filter(out, r1, n_loc1, n1);
    dscal(r1, -1.0, size1);
    daxpy(in, r1, 1.0, size1);
    restriction(r1, r2, n_loc1, n1, n_start1);  // r2 = restriction(r1)

    smooth(r2, e2, n_loc2, n2, sm2);  // e2 = smooth(r2)  ~solve(A . e2 = r2)
    // tmp2  =  r2 - A . e2
    laplace_filter(e2, tmp2, n_loc2, n2);
    dscal(tmp2, -1.0, size2);
    daxpy(r2, tmp2, 1.0, size2);
    restriction(tmp2, r3, n_loc2, n2, n_start2);  // r3 = restriction(r2 - A . e2)


    smooth(r3, e3, n_loc3, n3, sm3);  // e3 = smooth(r3)  ~solve(A . e3 = r3)
    prolong(e3, r2, n_loc3, n3, n_loc2, n2, n_start2);
    daxpy(r2, e2, 1.0, size2);  // e2 = e2 + prolong(e3)

    smooth(r2, e2, n_loc2, n2, sm2);  // e2 = smooth(r2, e2)  ~solve(A . e2 = r2)
    prolong(e2, r1, n_loc2, n2, n_loc1, n1, n_start1);
    daxpy(r1, out, 1.0, size1);  // out = out + prolong(e2)

    smooth(in, out, n_loc1, n1, sm4);  // out = smooth(in, out)  ~solve(A . out = in)

    mpi_grid_free(r1, n1);
    mpi_grid_free(r2, n2);
    mpi_grid_free(r3, n3);
    mpi_grid_free(e2, n2);
    mpi_grid_free(e3, n3);
    mpi_grid_free(tmp2, n2);
}

/*
Apply the multigrid method to solve the Poisson equation  A.out = in
using a 3-level V-cycle multigrid method.

@param in: input array (right-hand side of the equation)
@param out: in/out array (starting guess/solution)
@param s1: size of the first dimension (number of slices)
@param s2: size of the second dimension (number of grid points per slice)
@param n_start1: starting index for the first dimension (used for restriction)
*/
void multigrid_apply_2lvl(
    double *in, double *out, int s1, int s2, int n_start1,
    int sm1, int sm2, int sm3, int sm4
) {
    int n1 = s2;
    int n2 = n1 / 2;

    int n_loc1 = s1;

    int n_loc2 = (n_loc1 + 1 - (n_start1 % 2)) / 2;
    int n_start2 = (n_start1 + 1) / 2;


    if (n_loc2 == 0) {
        mpi_fprintf(stderr, "------------------------------------------------------------------------------------\n");
        mpi_fprintf(stderr, "Warning: after restriction some processors have no local grid points!\n");
        mpi_fprintf(stderr, "This case is not yet implemented, please use MULTIGRID with atleast 4 slices\n");
        mpi_fprintf(stderr, "per processor (N_grid / num_mpi_procs >= 4) \n");
        mpi_fprintf(stderr, "------------------------------------------------------------------------------------\n");
        exit(1);
    }

    long int size1 = n_loc1 * n1 * n1;
    long int size2 = n_loc2 * n2 * n2;

    double *r1 = mpi_grid_allocate(n_loc1, n1);
    double *r2 = mpi_grid_allocate(n_loc2, n2);

    double *e2 = mpi_grid_allocate(n_loc2, n2);

    double *tmp2 = mpi_grid_allocate(n_loc2, n2);

    // memset(out, 0, size1 * sizeof(double));  // out = 0
    memset(e2, 0, size2 * sizeof(double));  // e2 = 0

    smooth(in, out, n_loc1, n1, sm1);  // out = smooth(in, out)  ~solve(A . out = in)
    // r1  =  in - A . out
    laplace_filter(out, r1, n_loc1, n1);
    dscal(r1, -1.0, size1);
    daxpy(in, r1, 1.0, size1);
    restriction(r1, r2, n_loc1, n1, n_start1);  // r2 = restriction(r1)


    smooth(r2, e2, n_loc2, n2, sm2);  // e2 = smooth(r2, e2)  ~solve(A . e2 = r2)
    prolong(e2, r1, n_loc2, n2, n_loc1, n1, n_start1);
    daxpy(r1, out, 1.0, size1);  // out = out + prolong(e2)

    smooth(in, out, n_loc1, n1, sm4);  // out = smooth(in, out)  ~solve(A . out = in)

    mpi_grid_free(r1, n1);
    mpi_grid_free(r2, n2);
    // mpi_grid_free(r3, n3);
    mpi_grid_free(e2, n2);
    // mpi_grid_free(e3, n3);
    mpi_grid_free(tmp2, n2);
}

void prolong_nearestneighbors(
    double *in, double *out, int s1, int s2,
    int target_s1, int target_s2, int target_n_start
) {
    int a, b;
    long int i, j, k;
    long int i0, j0, k0;
    long int i1, j1, k1;
    long int n2 = s2 * s2;

    long int target_n2 = target_s2 * target_s2;

    double app;

    int d = target_n_start % 2;

    #pragma omp parallel for private(i, j, k, i0, j0, k0, i1, j1, k1, a, b, app)
    for (i = 0; i < s1; i++) {
        a = i * n2;
        i0 = (i * 2 + d) * target_n2;
        i1 = i0 + target_n2;
        for (j = 0; j < s2; j++) {
            b = j * s2;
            j0 = j * 2 * target_s2;
            j1 = j0 + target_s2;
            for (k = 0; k < s2; k++) {
                k0 = k * 2;
                k1 = k0 + 1;
                app = in[a + b + k] * 0.125;
                out[i0 + j0 + k0] = app;
                out[i0 + j0 + k1] = app;
                out[i0 + j1 + k0] = app;
                out[i0 + j1 + k1] = app;
                out[i1 + j0 + k0] = app;
                out[i1 + j0 + k1] = app;
                out[i1 + j1 + k0] = app;
                out[i1 + j1 + k1] = app;
            }
        }
    }

    // In case of odd number of slices, we need to wrap around the top slice from the proc below
    // as the 1st bottom slice of the current proc
    mpi_grid_exchange_bot_top(out, target_s1, target_s2);  // Called outside the if to avoid deadlock
    if (d) {
        vec_copy(out - target_n2, out, target_n2);
    }
}

/*
Apply trilinear interpolation prolongation to transfer corrections
from a coarse grid (s1 x s2 x s2) to a fine grid (target_s1 x target_s2 x target_s2).
Each fine node is computed once using the 8 neighboring coarse nodes.
Periodic BCs are applied in j,k. Along i we rely on a single halo exchange
to read I+1 on rank boundaries. No writes to halo memory are performed here.

@param in:            input coarse-grid array
@param out:           output fine-grid array (overwritten; interior only)
@param s1:            coarse number of slices in i
@param s2:            coarse number of points per line in j,k
@param target_s1:     fine number of slices in i
@param target_s2:     fine number of points per line in j,k
@param target_n_start: starting index (global) for the fine grid in i (parity across MPI ranks)
*/
void prolong_trilinear(
    double *in, double *out, int s1, int s2,
    int target_s1, int target_s2, int target_n_start
) {
    long int n2c = s2 * s2;
    long int n2f = target_s2 * target_s2;

    long int row_f;
    long int i0, j0, k0;
    long int i1, j1, k1;
    double wi0, wj0, wk0;
    double wi1, wj1, wk1;
    // double c000, c100, c010, c110, c001, c101, c011, c111;

    int d = target_n_start & 1;

    // halo on coarse input
    mpi_grid_exchange_bot_top(in, s1, s2);

    // periodic neighbors for coarse j,k
    int *jnext = (int *)malloc(s2 * sizeof(int));
    // int *knext = (int *)malloc(s2 * sizeof(int));
    for (int t = 0; t < s2; t++) {
        jnext[t] = (t + 1 == s2) ? 0 : (t + 1);
        // knext[t] = jnext[t];
    }

    #pragma omp parallel for private(i0, j0, k0, i1, j1, k1, wi0, wj0, wk0, wi1, wj1, wk1, row_f)
    for (int i = 0; i < target_s1; i++) {
        int I0 = (i < d) ? (s1 - 1) : ((i - d) >> 1);
        i0 = I0 * n2c;
        i1 = i0 + n2c;

        if ((i + target_n_start) % 2) {
            wi0 = 0.5;
            wi1 = 0.5;
        } else {
            wi0 = 1.0;
            wi1 = 0.0;
        }

        for (int j = 0; j < target_s2; j++) {
            // int J0 = j / 2;
            // int J1 = jnext[J0];
            // int J1 = (j + 1) % s2;
            j0 = ((j     + 1) % s2) * s2;
            j1 = ((j / 2 + 1) % s2) * s2;

            if (j % 2) {
                wj0 = 1.0;
                wj1 = 0.0;
            } else {
                wj0 = 0.5;
                wj1 = 0.5;
            }

            row_f = i * n2f + j * target_s2;

            for (int k = 0; k < target_s2; k++) {
                k0 = k / 2;
                k1 = jnext[k0];

                if (k % 2) {
                    wk0 = 1.0;
                    wk1 = 0.0;
                } else {
                    wk0 = 0.5;
                    wk1 = 0.5;
                }

                out[row_f + k] =
                    in[i0 + j0 + k0] * wi0 * wj0 * wk0 +
                    in[i1 + j0 + k0] * wi1 * wj0 * wk0 +
                    in[i0 + j1 + k0] * wi0 * wj1 * wk0 +
                    in[i1 + j1 + k0] * wi1 * wj1 * wk0 +
                    in[i0 + j0 + k1] * wi0 * wj0 * wk1 +
                    in[i1 + j0 + k1] * wi1 * wj0 * wk1 +
                    in[i0 + j1 + k1] * wi0 * wj1 * wk1 +
                    in[i1 + j1 + k1] * wi1 * wj1 * wk1;
            }
        }
    }

    free(jnext);
    // free(knext);

    // legacy post-condition: out halos ready + wrap if d==1
    mpi_grid_exchange_bot_top(out, target_s1, target_s2);
    if (d) {
        vec_copy(out - n2f, out, n2f);
    }
}


void restriction_8pt(double *in, double *out, int s1, int s2, int n_start) {
    int a, b;
    long int i, j, k;
    long int i0, j0;
    long int i1, j1, k1;
    long int n2 = s2 * s2;

    int s3 = s2 / 2;
    long int n3 = s3 * s3;

    // If the number of slices in the first dimension is odd, we need to wrap around
    // the bottom slice above to apply the averaging with PBCs
    mpi_grid_exchange_bot_top(in, s1, s2);

    #pragma omp parallel for private(i, j, k, i0, i1, j0, j1, k1, a, b)
    for (i = n_start % 2; i < s1; i+=2) {
        i0 = i * n2;
        i1 = (i+1) * n2;
        a = i / 2 * n3;
        for (j = 0; j < s2; j+=2) {
            j0 = j * s2;
            j1 = ((j+1) % s2) * s2;
            b = j / 2 * s3;
            for (k = 0; k < s2; k+=2) {
                k1 = (k+1) % s2;
                out[a + b + k / 2] = (
                    in[i0 + j0 + k] +
                    in[i0 + j0 + k1] +
                    in[i0 + j1 + k] +
                    in[i0 + j1 + k1] +
                    in[i1 + j0 + k] +
                    in[i1 + j0 + k1] +
                    in[i1 + j1 + k] +
                    in[i1 + j1 + k1]
                ) * 0.125;
            }
        }
    }
}

/*
Apply 27-point full-weighting restriction (nodal) to transfer residuals
from a fine grid (s1 x s2 x s2) to a coarse grid (s1/2 x s2/2 x s2/2).
The coarse value at (I,J,K) corresponds to the fine node (i=2I+offset, j=2J, k=2K),
and is computed with weights: center=8, faces=4, edges=2, corners=1, normalized by 64.

@param in:       input fine-grid array (e.g., residual on fine grid)
@param out:      output coarse-grid array (restricted residual)
@param s1:       number of slices in i (fine grid)
@param s2:       number of points per line in j,k (fine grid)
@param n_start:  global starting index in i (used to preserve parity across MPI ranks)
*/
void restriction_27pt(double *in, double *out, int s1, int s2, int n_start) {
    int s3 = s2 / 2;
    long int n2 = s2 * s2;
    long int n3 = s3 * s3;
    double inv64 = 1.0 / 64.0;

    // legacy behavior: ensure we can read i-1 / i+1 from contiguous ghosts
    mpi_grid_exchange_bot_top(in, s1, s2);

    int *jprev = (int *)malloc(s2 * sizeof(int));
    int *jnext = (int *)malloc(s2 * sizeof(int));
    // int *kprev = (int *)malloc(s2 * sizeof(int));
    // int *knext = (int *)malloc(s2 * sizeof(int));
    for (int t = 0; t < s2; t++) {
        jprev[t] = (t == 0)      ? (s2 - 1) : (t - 1);
        jnext[t] = (t + 1 == s2) ? 0        : (t + 1);
        // kprev[t] = jprev[t];
        // knext[t] = jnext[t];
    }

    int i_offset = n_start & 1;

    #pragma omp parallel for
    for (int i = i_offset; i < s1; i += 2) {
        long int i0 = i * n2;
        long int im = i0 - n2;      // contiguous ghost
        long int ip = i0 + n2;      // contiguous ghost
        long int a  = (i / 2) * n3;

        for (int j = 0; j < s2; j += 2) {
            long int j0  = j * s2;
            long int jpm = jprev[j] * s2;
            long int jpp = jnext[j] * s2;
            long int b   = (j / 2) * s3;

            for (int k = 0; k < s2; k += 2) {
                int km = jprev[k];
                int kp = jnext[k];

                long int idx_c   = i0 + j0 + k;
                long int out_idx = a + b + (k >> 1);

                double f_sum =
                    in[im + j0 + k] + in[ip + j0 + k] +
                    in[i0 + jpm + k] + in[i0 + jpp + k] +
                    in[i0 + j0 + km] + in[i0 + j0 + kp];

                double e_sum =
                    in[im + jpm + k] + in[im + jpp + k] +
                    in[ip + jpm + k] + in[ip + jpp + k] +
                    in[im + j0 + km] + in[im + j0 + kp] +
                    in[ip + j0 + km] + in[ip + j0 + kp] +
                    in[i0 + jpm + km] + in[i0 + jpm + kp] +
                    in[i0 + jpp + km] + in[i0 + jpp + kp];

                double c_sum =
                    in[im + jpm + km] + in[im + jpm + kp] +
                    in[im + jpp + km] + in[im + jpp + kp] +
                    in[ip + jpm + km] + in[ip + jpm + kp] +
                    in[ip + jpp + km] + in[ip + jpp + kp];

                out[out_idx] = (8.0 * in[idx_c] + 4.0 * f_sum + 2.0 * e_sum + c_sum) * inv64;
            }
        }
    }

    free(jprev);
    free(jnext);
    // free(kprev);
    // free(knext);
}


/*
Apply the Jacobi smoothing method to solve the Poisson equation.
Gives an approximate solution to the equation A.out = in based

@param in: input array (right-hand side of the equation)
@param out: in/out array (starting guess/solution)
@param s1: size of the first dimension (number of slices)
@param s2: size of the second dimension (number of grid points per slice)
@param tol: number of iterations to perform
*/
void smooth_jacobi(double *in, double *out, int s1, int s2, double tol) {
    long int n3 = s1 * s2 * s2;

    double omega = JACOBI_OMEGA / -6.0;

    double *tmp = (double *)malloc(n3 * sizeof(double));

    for (int iter=0; iter < tol; iter++) { 
        // out = out + omega * (in - A. out)
        laplace_filter(out, tmp, s1, s2);  // res = A . out
        daxpy(tmp, out, -omega, n3);
        daxpy(in, out, omega, n3);
    }

    free(tmp);
}


/*
Apply the Red-Black Gauss-Seidel smoothing method to solve the Poisson equation.
Updates the solution in-place, giving an approximate solution to A . out = in
with a 7-point Laplacian stencil under periodic boundary conditions in j,k.

@param in:   input array (right-hand side of the equation)
@param out:  in/out array (initial guess and updated solution)
@param s1:   size of the first dimension (number of slices in i)
@param s2:   size of the second dimension (number of grid points in j,k)
@param tol:  number of smoothing iterations (each iteration = red + black sweep)
*/
void smooth_rbgs(double *in, double *out, int s1, int s2, double tol) {
    int iters = (int)tol;
    if (iters <= 0) {
        return;
    }

    long int n2 = s2 * s2;
    double inv_diag = -1.0 / 6.0;   // diagonal entry of 7-point Laplacian
    int n_start = get_n_start();    // global offset for MPI parity

    // Precompute neighbor indices for periodic BCs in j and k
    int *jprev = (int *)malloc((size_t)s2 * sizeof(int));
    int *jnext = (int *)malloc((size_t)s2 * sizeof(int));
    int *kprev = (int *)malloc((size_t)s2 * sizeof(int));
    int *knext = (int *)malloc((size_t)s2 * sizeof(int));
    for (int t = 0; t < s2; t++) {
        jprev[t] = (t == 0)      ? (s2 - 1) : (t - 1);
        jnext[t] = (t + 1 == s2) ? 0        : (t + 1);
        kprev[t] = jprev[t];
        knext[t] = jnext[t];
    }

    for (int iter = 0; iter < iters; iter++) {
        // Exchange ghost slices before starting the red sweep
        mpi_grid_exchange_bot_top(out, s1, s2);

        // ----- RED sweep -----
        #pragma omp parallel for
        for (int i = 0; i < s1; i++) {
            long int i0 = i * n2;
            long int im = i0 - n2;      // ghost if i==0
            long int ip = i0 + n2;      // ghost if i==s1-1
            int par_i = (n_start + i) & 1;

            for (int j = 0; j < s2; j++) {
                long int j0  = j * s2;
                long int jpm = jprev[j] * s2;
                long int jpp = jnext[j] * s2;

                // choose k parity so that (i+j+k) % 2 == 0 (red)
                int k = (2 - ((par_i + (j & 1)) & 1)) & 1;
                for (; k < s2; k += 2) {
                    int km = kprev[k];
                    int kp = knext[k];
                    long int idx = i0 + j0 + k;

                    double sum_nb =
                        out[im + j0 + k] + out[ip + j0 + k] +     // i-1, i+1
                        out[i0 + jpm + k] + out[i0 + jpp + k] +   // j-1, j+1
                        out[i0 + j0  + km] + out[i0 + j0  + kp];  // k-1, k+1

                    out[idx] = (in[idx] - sum_nb) * inv_diag;
                }
            }
        }

        // Exchange halo again before the black sweep
        mpi_grid_exchange_bot_top(out, s1, s2);

        // ----- BLACK sweep -----
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < s1; i++) {
            long int i0 = i * n2;
            long int im = i0 - n2;
            long int ip = i0 + n2;
            int par_i = (n_start + i) & 1;

            for (int j = 0; j < s2; j++) {
                long int j0  = j * s2;
                long int jpm = jprev[j] * s2;
                long int jpp = jnext[j] * s2;

                // choose k parity so that (i+j+k) % 2 == 1 (black)
                int k = (1 - ((par_i + (j & 1)) & 1)) & 1;
                for (; k < s2; k += 2) {
                    int km = kprev[k];
                    int kp = knext[k];
                    long int idx = i0 + j0 + k;

                    double sum_nb =
                        out[im + j0 + k] + out[ip + j0 + k] +
                        out[i0 + jpm + k] + out[i0 + jpp + k] +
                        out[i0 + j0  + km] + out[i0 + j0  + kp];

                    out[idx] = (in[idx] - sum_nb) * inv_diag;
                }
            }
        }
        // After black sweep: next iteration will exchange halos again
    }

    free(jprev);
    free(jnext);
    free(kprev);
    free(knext);
}


// void smooth_lcg(double *in, double *out, int s1, int s2, double tol) {
//     conj_grad(in, out, out, 1E-1, s1, s2);  // out = ~solve(A . out = in)
// }


// void smooth_diag(double *in, double *out, int s1, int s2, double tol) {
//     long int n3 = s1 * s2 * s2;
//     for (int i = 0; i < n3; i++) {
//         out[i] = in[i] / -6.0;
//     }
// }


// choose smoothing, restriction and interpolation
void smooth(double *in, double *out, int s1, int s2, double tol) {
    // smooth_lcg(in, out, s1, s2, tol);
    // smooth_jacobi(in, out, s1, s2, tol);
    // smooth_diag(in, out, s1, s2, tol);
    smooth_rbgs(in, out, s1, s2, tol);
}

void restriction(double *in, double *out, int s1, int s2, int n_start) {
    // restriction_8pt(in, out, s1, s2, n_start);
    restriction_27pt(in, out, s1, s2, n_start);
}

void prolong(double *in, double *out, int s1, int s2, int target_s1, int target_s2, int target_n_start) {
    // prolong_nearestneighbors(in, out, s1, s2, target_s1, target_s2, target_n_start);
    prolong_trilinear(in, out, s1, s2, target_s1, target_s2, target_n_start);
}
