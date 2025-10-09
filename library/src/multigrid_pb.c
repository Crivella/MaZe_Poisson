#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "linalg.h"
#include "verlet.h"
#include "mp_structs.h"
#include "mpi_base.h"

#define JACOBI_OMEGA 0.66
#define AXIS_X 0
#define AXIS_Y 1
#define AXIS_Z 2

static int cg_coarse_pb(double* b, double* x, int s1, int s2, int maxit, double rtol,
                        double *eps_x, double *eps_y, double *eps_z, double *k2_screen)
{
    const long n = (long)s1 * (long)s2 * (long)s2;

    /* --- scalars --- */
    int    k;
    double r0_inf, r_inf;
    double r_dot_r, denom, alpha;
    double rn_rn, beta;

    /* --- buffers with ghost --- */
    double *r;
    double *p;
    double *Ap;

    /* --- allocations --- */
    r  = mpi_grid_allocate(s1, s2);
    p  = mpi_grid_allocate(s1, s2);
    Ap = mpi_grid_allocate(s1, s2);

    /* r = A x - b  (x is the initial guess provided by the caller) */
    laplace_filter_pb(x, r, s1, s2, eps_x, eps_y, eps_z, k2_screen);   /* r <- A x */
    daxpy(b, r, -1.0, n);                                             /* r <- r - b */

    /* p = -r (no preconditioner) */
    memcpy(p, r, n * sizeof(double));
    dscal(p, -1.0, n);

    /* initial norm (relative stop criterion) */
    r0_inf = norm_inf(r, n);
    if (r0_inf == 0.0) {
        mpi_grid_free(r, s2);
        mpi_grid_free(p, s2);
        mpi_grid_free(Ap, s2);
        return 0;
    }

    /* r_dot_r = <r, r> */
    r_dot_r = ddot(r, r, n);

    for (k = 0; k < maxit; ++k) {
        /* Ap = A p */
        laplace_filter_pb(p, Ap, s1, s2, eps_x, eps_y, eps_z, k2_screen);

        denom = ddot(p, Ap, n);
        if (fabs(denom) < 1e-300) {
            /* almost zero direction / breakdown */
            break;
        }

        alpha = r_dot_r / denom;

        /* x <- x + alpha p */
        daxpy(p, x, alpha, n);

        /* r <- r + alpha Ap   (since r = A x - b) */
        daxpy(Ap, r, alpha, n);

        /* stopping criterion (relative infinity norm) */
        r_inf = norm_inf(r, n);
        // printf("cg iter=%d \t res=%e\n", k+1, r_inf);
        if (r_inf <= rtol * r0_inf) {
            ++k;
            break;
        }

        /* update scalars */
        rn_rn = ddot(r, r, n);
        beta  = rn_rn / r_dot_r;
        r_dot_r = rn_rn;

        /* p <- -r + beta p */
        dscal(p, beta, n);
        daxpy(r, p, -1.0, n);
    }

    mpi_grid_free(r,  s2);
    mpi_grid_free(p,  s2);
    mpi_grid_free(Ap, s2);

    return k;  /* number of iterations executed */
}

static int cg_coarse_pb_jacobi(double* b, double* x, int s1, int s2, int maxit, double rtol,
                               double *eps_x, double *eps_y, double *eps_z, double *k2_screen)
{
    const long n = (long)s1 * (long)s2 * (long)s2;

    /* --- scalars --- */
    int    k;
    double r0_inf, r_inf;
    double denom, alpha;
    double rho, rho_new, beta;

    /* --- buffers with ghost --- */
    double *r;      /* residual r = A x - b */
    double *p;      /* search direction */
    double *Ap;     /* A p */
    double *D;      /* Jacobi diagonal (positive) */
    double tiny;    /* safeguard for division */

    /* --- neighbor wrap arrays for periodic j,k --- */
    int *jnext;
    int *knext;

    /* --- indices --- */
    long int i, j, k3;
    long int i0, i1, i2;
    long int j0, j1;
    long int k0, k1i;
    long int idx0, idx_xm, idx_ym, idx_zm;

    /* --- temps for diagonal --- */
    double ex_p, ex_m, ey_p, ey_m, ez_p, ez_m, diag_pos;

    /* --- sizes --- */
    const long n2 = (long)s2 * (long)s2;

    /* --- allocations --- */
    r   = mpi_grid_allocate(s1, s2);
    p   = mpi_grid_allocate(s1, s2);
    Ap  = mpi_grid_allocate(s1, s2);
    D   = mpi_grid_allocate(s1, s2);  /* same shape as field */

    jnext = (int*)malloc((size_t)s2 * sizeof(int));
    knext = (int*)malloc((size_t)s2 * sizeof(int));

    tiny = 1e-30;

    /* periodic wrap for j,k (only +1 needed here) */
    for (int t = 0; t < s2; ++t) {
        knext[t] = (t + 1) % s2;
        jnext[t] = knext[t] * s2;
    }

    /* halo for eps_* (we read i-1 plane for ex_m, ey_m, ez_m via idx_xm/ym/zm) */
    mpi_grid_exchange_bot_top(eps_x, s1, s2);
    mpi_grid_exchange_bot_top(eps_y, s1, s2);
    mpi_grid_exchange_bot_top(eps_z, s1, s2);

    /* ---- build Jacobi diagonal D = sum(face coeffs) + k2 (positive) ---- */
    #pragma omp parallel for private(i0,i1,i2,j0,j1,k0,k1i,idx0,idx_xm,idx_ym,idx_zm,ex_p,ex_m,ey_p,ey_m,ez_p,ez_m,diag_pos)
    for (i = 0; i < s1; ++i) {
        i0 = i * n2;       /* i plane */
        i1 = i0 + n2;      /* i+1 (ghost allowed) */
        i2 = i0 - n2;      /* i-1 (ghost allowed) */
        for (j = 0; j < s2; ++j) {
            j0 = j * s2;       /* j */
            j1 = jnext[j];     /* j+1 (periodic) */
            for (k3 = 0; k3 < s2; ++k3) {
                k0   = k3;             /* k */
                k1i  = knext[k3];      /* k+1 (periodic) */

                idx0   = i0 + j0 + k0;         /* center cell */
                idx_xm = i2 + j0 + k0;         /* (i-1,j,k)  -> face i-1/2 */
                idx_ym = i0 + j0 - s2 + k0;    /* (i,j-1,k) -> jprev: j0 - s2 */
                idx_zm = i0 + j0 + ((k3 - 1 + s2) % s2); /* (i,j,k-1) */

                /* face-centered coeffs at +dir from center and -dir from neighbor */
                ex_p = eps_x[idx0];         /* (i+1/2) */
                ex_m = eps_x[idx_xm];       /* (i-1/2) */
                ey_p = eps_y[idx0];         /* (j+1/2) */
                ey_m = eps_y[i0 + j0 + k0 - 0]; /* will fix below */

                /* correct ey_m and ez_m using precomputed indices */
                ey_m = eps_y[idx_ym];                   /* (j-1/2) */
                ez_p = eps_z[idx0];                     /* (k+1/2) */
                ez_m = eps_z[i0 + j0 + ((k3 - 1 + s2) % s2)]; /* (k-1/2) == eps_z[idx_zm] */

                diag_pos = ex_p + ex_m + ey_p + ey_m + ez_p + ez_m + k2_screen[idx0];
                D[idx0] = diag_pos;
            }
        }
    }

    /* r = A x - b */
    laplace_filter_pb(x, r, s1, s2, eps_x, eps_y, eps_z, k2_screen);
    daxpy(b, r, -1.0, n);

    /* initial norm (relative stop) */
    r0_inf = norm_inf(r, n);
    if (r0_inf == 0.0) {
        mpi_grid_free(r,  s2);
        mpi_grid_free(p,  s2);
        mpi_grid_free(Ap, s2);
        mpi_grid_free(D,  s2);
        free(jnext); free(knext);
        return 0;
    }

    /* z = - M^{-1} r  with M = D (Jacobi), D>0 */
    #pragma omp parallel for
    for (long t = 0; t < n; ++t) {
        double d = D[t];
        double inv = (d > tiny) ? (1.0 / d) : 0.0;
        /* minus sign matches r = A x - b so that with M=I we get z = -r */
        p[t] = - r[t] * inv;   /* reuse p as z initially */
    }

    /* rho = r^T z */
    rho = ddot(r, p, n);

    /* initial search direction p <- z (already in p) */

    for (k = 0; k < maxit; ++k) {
        /* Ap = A p */
        laplace_filter_pb(p, Ap, s1, s2, eps_x, eps_y, eps_z, k2_screen);

        denom = ddot(p, Ap, n);
        if (fabs(denom) < 1e-300) {
            /* near-breakdown */
            break;
        }

        alpha = rho / denom;

        /* x <- x + alpha p */
        daxpy(p, x, alpha, n);

        /* r <- r + alpha Ap   (since r = A x - b) */
        daxpy(Ap, r, alpha, n);

        /* check convergence (relative infinity norm) */
        r_inf = norm_inf(r, n);
        if (r_inf <= rtol * r0_inf) {
            ++k;
            break;
        }

        /* z_new = - M^{-1} r */
        #pragma omp parallel for
        for (long t = 0; t < n; ++t) {
            double d = D[t];
            double inv = (d > tiny) ? (1.0 / d) : 0.0;
            Ap[t] = - r[t] * inv;   /* reuse Ap as z_new temporarily */
        }

        rho_new = ddot(r, Ap, n);
        beta    = rho_new / rho;
        rho     = rho_new;

        /* p <- z_new + beta p  (z_new currently in Ap) */
        dscal(p, beta, n);
        daxpy(Ap, p, 1.0, n);
    }

    mpi_grid_free(r,  s2);
    mpi_grid_free(p,  s2);
    mpi_grid_free(Ap, s2);
    mpi_grid_free(D,  s2);
    free(jnext); free(knext);

    return k;  /* number of iterations executed */
}


// Recursive V-cycle for PB equation: smoothing is adapted for PB, restriction and prolongation are the same as for Poisson 
// + restriction for epsilon (face-centered)
int v_cycle_pb(double *in, double *out, int s1, int s2, int n_start, int sm, int depth, double *eps_x, double *eps_y, double *eps_z, double *k2_screen) {
    int res;
    const long int n2   = (long int)s2 * s2;
    const long int size = (long int)s1 * n2;

    // next level sizes/parity
    const int s1_nxt      = (s1 + 1 - (n_start % 2)) / 2;
    const int s2_nxt      = s2 / 2;

    // Se lo giri senza MPI funziona anche senza il +1 ma con il +1 server per tenero conto di quando n_start e' dispari
    const int n_start_nxt = (n_start + 1) / 2;   // CHANGED: floor  

    const int sm_iter = (int)ceil(sm * pow(MG_RECURSION_FACTOR, depth));

    // base case
    if ( (s1_nxt < fmax(16, get_size())) || (depth >= 1) ) {
        if (depth == 0) {
            mpi_fprintf(stderr, "------------------------------------------------------------------------------------\n");
            mpi_fprintf(stderr, "Multigrid: requires atleast one level of recursion (s1 >= %d)\n", fmax(4, get_size()));
            mpi_fprintf(stderr, "Increase the size of the grid or reduce the number of MPI processes.\n");
            mpi_fprintf(stderr, "------------------------------------------------------------------------------------\n");
            exit(1);
        }
        cg_coarse_pb(in, out, s1, s2, 50, 1e-5, eps_x, eps_y, eps_z, k2_screen); 
        return depth;
    }

    const long int size_nxt = (long int)s1_nxt * s2_nxt * s2_nxt;

    // buffers
    double *r   = mpi_grid_allocate(s1, s2);
    double *rhs = mpi_grid_allocate(s1_nxt, s2_nxt);
    double *eps = mpi_grid_allocate(s1_nxt, s2_nxt);
    double *eps_x_c = mpi_grid_allocate(s1_nxt, s2_nxt);
    double *eps_y_c = mpi_grid_allocate(s1_nxt, s2_nxt);
    double *eps_z_c = mpi_grid_allocate(s1_nxt, s2_nxt);
    double *k2_screen_c = mpi_grid_allocate(s1_nxt, s2_nxt);

    memset(eps, 0, size_nxt * sizeof(double));

    // 1) pre-smoothing
    smooth_pb(in, out, s1, s2, sm_iter, eps_x, eps_y, eps_z, k2_screen);

    // 2) residual: r = in - A*out
    laplace_filter_pb(out, r, s1, s2, eps_x, eps_y, eps_z, k2_screen);
    dscal(r, -1.0, size);
    daxpy(in, r, 1.0, size);

    // 3) restrict to coarse
    restriction(r, rhs, s1, s2, n_start);
    
    // Restrict epsilon arrays to coarse grid
    restriction_eps(eps_x, eps_x_c, s1, s2, AXIS_X);
    restriction_eps(eps_y, eps_y_c, s1, s2, AXIS_Y);
    restriction_eps(eps_z, eps_z_c, s1, s2, AXIS_Z);

    // Restrict k2_screen to coarse grid
    restriction_k2screen(k2_screen, k2_screen_c, s1, s2, n_start); 

    // 4) coarse solve (recursive)
    res = v_cycle_pb(rhs, eps, s1_nxt, s2_nxt, n_start_nxt, sm, depth + 1, eps_x_c, eps_y_c, eps_z_c, k2_screen_c);

    // 5) prolong correction
    prolong(eps, r, s1_nxt, s2_nxt, s1, s2, n_start);

    // 6) apply correction
    daxpy(r, out, 1.0, size);

    // 7) post-smoothing
    smooth_pb(in, out, s1, s2, sm_iter, eps_x, eps_y, eps_z, k2_screen);

    mpi_grid_free(r, s2);
    mpi_grid_free(rhs, s2_nxt);
    mpi_grid_free(eps, s2_nxt);
    mpi_grid_free(eps_x_c, s2_nxt);
    mpi_grid_free(eps_y_c, s2_nxt);
    mpi_grid_free(eps_z_c, s2_nxt);
    mpi_grid_free(k2_screen_c, s2_nxt);

    return res;
}


int multigrid_pb_apply_recursive(double *in, double *out, int s1, int s2, int n_start1, int sm, double *eps_x, double *eps_y, double *eps_z, double *k2_screen) {
    // memset(out, 0, s1 * s2 * s2 * sizeof(double));  // Initialize out to zero
    return v_cycle_pb(in, out, s1, s2, n_start1, sm, 0, eps_x, eps_y, eps_z, k2_screen);  // Apply the recursive V-cycle multigrid method
}

/*
Restrict face-centered coefficient eps (stored as face in +axis direction of each cell)
from fine grid (s1 x s2 x s2) to coarse grid (s1/2 x s2/2 x s2/2)
by 2x2 area averaging on the plane orthogonal to 'axis'.

Axis encoding:
  axis = 0 -> X faces (JK plane average, uses i plane only)
  axis = 1 -> Y faces (IK plane average, uses i and i+1 planes)
  axis = 2 -> Z faces (IJ plane average, uses i and i+1 planes)

Output eps_H(I,J,K) corresponds to the coarse face in +axis direction of coarse cell (I,J,K).
*/
void restriction_eps(double *eps_in, double *eps_out, int s1, int s2, int axis) {
    int s3;
    long int n2, n3;

    long int I, J, K;
    long int i, j, k;
    long int i0, i1;
    long int j0, j1;
    long int k0, k1;
    long int a, b, c;

    double sum4, inv4;

    int jnext[s2];
    int knext[s2];

    s3  = s2 / 2;
    n2  = (long)s2 * (long)s2;
    n3  = (long)s3 * (long)s3;
    inv4 = 0.25;

    // Precompute neighbor indices for periodic BCs in j and k
    for (int t = 0; t < s2; ++t) {
        knext[t] = ((t + 1) % s2);
        jnext[t] = knext[t] * s2;
    }

    // For AXIS_Y and AXIS_Z, caller must ensure i-direction halo is available in 'eps_in'
    if (axis != AXIS_X) {
        mpi_grid_exchange_bot_top(eps_in, s1, s2);
    }

    if (axis == AXIS_X) {
        // 2x2 average on JK plane at (i+1/2, j, k)
        #pragma omp parallel for private(I,J,K,i,j,k,i0,j0,j1,k0,k1,a,b,c,sum4)
        for (I = 0; I < s1/2; ++I) {
            i  = 2*I;                 // fine i plane
            i0 = (long)i * n2;        // base plane at i
            a  = (long)I * n3;        // coarse i offset

            for (J = 0; J < s2/2; ++J) {
                j  = 2*J;
                j0 = (long)j * s2;    // j
                j1 = jnext[j];        // j+1
                b  = (long)J * s3;    // coarse j offset

                for (K = 0; K < s2/2; ++K) {
                    k  = 2*K;
                    k0 = k;            // k
                    k1 = knext[k];     // k+1
                    c  = (long)K;

                    sum4 =
                        eps_in[i0 + j0 + k0] +
                        eps_in[i0 + j0 + k1] +
                        eps_in[i0 + j1 + k0] +
                        eps_in[i0 + j1 + k1];

                    eps_out[a + b + c] = sum4 * inv4;
                }
            }
        }
    } else if (axis == AXIS_Y) {
        // 2x2 average on IK plane at (i, j+1/2, k) -> needs i and i+1
        #pragma omp parallel for private(I,J,K,i,j,k,i0,i1,j0,k0,k1,a,b,c,sum4)
        for (I = 0; I < s1/2; ++I) {
            i  = 2*I;
            i0 = (long)i * n2;        // i
            i1 = i0 + n2;             // i+1
            a  = (long)I * n3;

            for (J = 0; J < s2/2; ++J) {
                j  = 2*J;
                j0 = (long)j * s2;    // fixed j plane for eps_y
                b  = (long)J * s3;

                for (K = 0; K < s2/2; ++K) {
                    k  = 2*K;
                    k0 = k;            // k
                    k1 = knext[k];     // k+1
                    c  = (long)K;

                    sum4 =
                        eps_in[i0 + j0 + k0] +
                        eps_in[i0 + j0 + k1] +
                        eps_in[i1 + j0 + k0] +
                        eps_in[i1 + j0 + k1];

                    eps_out[a + b + c] = sum4 * inv4;
                }
            }
        }
    } else { /* AXIS_Z */
        // 2x2 average on IJ plane at (i, j, k+1/2) -> needs i and i+1
        #pragma omp parallel for private(I,J,K,i,j,k,i0,i1,j0,j1,k0,a,b,c,sum4)
        for (I = 0; I < s1/2; ++I) {
            i  = 2*I;
            i0 = (long)i * n2;        // i
            i1 = i0 + n2;             // i+1
            a  = (long)I * n3;

            for (J = 0; J < s2/2; ++J) {
                j  = 2*J;
                j0 = (long)j * s2;    // j
                j1 = jnext[j];        // j+1
                b  = (long)J * s3;

                for (K = 0; K < s2/2; ++K) {
                    k  = 2*K;
                    k0 = k;            // fixed k plane for eps_z
                    c  = (long)K;

                    sum4 =
                        eps_in[i0 + j0 + k0] +
                        eps_in[i1 + j0 + k0] +
                        eps_in[i0 + j1 + k0] +
                        eps_in[i1 + j1 + k0];

                    eps_out[a + b + c] = sum4 * inv4;
                }
            }
        }
    }
}


/*
 * Restriction per campi cell-centered (es. k2_screen).
 * Fine:  s1 x s2 x s2
 * Coarse:(s1/2) x (s2/2) x (s2/2)
 * - PBC in i, j, k tramite tabelle precompute (no ghost richiesti).
 * - Offset cell-centered: i_offset = 1 - (n_start % 2)
 * - s1 e s2 devono essere pari.
 */
void restriction_k2screen(const double *in, double *out, int s1, int s2, int n_start)
{
    const int s3  = s2 / 2;
    const long n2 = (long)s2 * s2;
    const long n3 = (long)s3 * s3;
    const double inv8 = 1.0 / 8.0;

    // Check veloci
    if ((s1 & 1) || (s2 & 1)) {
        fprintf(stderr, "restriction_k2screen: s1 (%d) e s2 (%d) devono essere pari\n", s1, s2);
        exit(1);
    }

    // Offset per campi cell-centered: complementare ai nodi
    const int i_offset = 1 - (n_start % 2);

    // Tabelle PBC per i (in stride di piani), j e k
    long iprev[s1], inext[s1];
    long jnext[s2];
    int  knext[s2];

    for (int i = 0; i < s1; ++i) {
        int im = (i - 1 + s1) % s1;
        int ip = (i + 1) % s1;
        iprev[i] = (long)im * n2;
        inext[i] = (long)ip * n2;
    }
    for (int t = 0; t < s2; ++t) {
        int tp = (t + 1) % s2;
        knext[t] = tp;
        jnext[t] = (long)tp * s2;
    }

    #pragma omp parallel for
    for (int i = i_offset; i < s1; i += 2) {
        long i0 = (long)i * n2;
        long i1 = inext[i];              // wrap periodico lungo i
        long a  = (long)(i / 2) * n3;

        for (int j = 0; j < s2; j += 2) {
            long j0 = (long)j * s2;
            long j1 = jnext[j];
            long b  = (long)(j / 2) * s3;

            for (int k = 0; k < s2; k += 2) {
                int  k0 = k;
                int  k1 = knext[k];
                long c  = k / 2;

                double sum =
                    in[i0 + j0 + k0] + in[i0 + j0 + k1] +
                    in[i0 + j1 + k0] + in[i0 + j1 + k1] +
                    in[i1 + j0 + k0] + in[i1 + j0 + k1] +
                    in[i1 + j1 + k0] + in[i1 + j1 + k1];

                out[a + b + c] = inv8 * sum;
            }
        }
    }
}

/*
Apply the Jacobi smoothing method to solve the PB equation.
Gives an approximate solution to the equation A.out = in based

@param in: input array (right-hand side of the equation)
@param out: in/out array (starting guess/solution)
@param s1: size of the first dimension (number of slices)
@param s2: size of the second dimension (number of grid points per slice)
@param tol: number of iterations to perform
@param eps_x, eps_y, eps_z: face-centered dielectric arrays
@param k2_screen: cell-centered screening coefficient array
*/
void smooth_pb_jacobi(double *in, double *out, int s1, int s2, double tol, double *eps_x, double *eps_y, double *eps_z, double *k2_screen) {
    long int n3 = s1 * s2 * s2;

    double omega = JACOBI_OMEGA / -6.0;

    double *tmp = (double *)malloc(n3 * sizeof(double));

    for (int iter=0; iter < tol; iter++) { 
        // out = out + omega * (in - A. out)
        laplace_filter_pb(out, tmp, s1, s2, eps_x, eps_y, eps_z, k2_screen);  // res = A . out
        daxpy(tmp, out, -omega, n3);
        daxpy(in, out, omega, n3);
    }

    free(tmp);
}


/*
Apply the Red-Black Gauss-Seidel smoothing method to solve the PB equation.
Updates the solution in-place, giving an approximate solution to A . out = in
with a 7-point Laplacian stencil under periodic boundary conditions in j,k.

@param in:   input array (right-hand side of the equation)
@param out:  in/out array (initial guess and updated solution)
@param s1:   size of the first dimension (number of slices in i)
@param s2:   size of the second dimension (number of grid points in j,k)
@param tol:  number of smoothing iterations (each iteration = red + black sweep)
@param eps_x, eps_y, eps_z: face-centered dielectric arrays
@param k2_screen: cell-centered screening coefficient array
*/
void smooth_pb_rbgs(
    double *in, double *out,
    int size1, int size2, double tol,
    double *eps_x, double *eps_y, double *eps_z, double *k2_screen
) {
    int iters, n_start, iter, t, i, j, k, d, k1, k2;
    long n2, i0, i1, i2, j0, j1, j2, idx0, idx_x, idx_y, idx_z;
    int *jprev, *jnext, *kprev, *knext;
    double DIAG_EPS, ex_p, ex_m, ey_p, ey_m, ez_p, ez_m, diag, rhs;

    iters = (int)tol; if (iters <= 0) return;
    n_start = get_n_start();
    n2 = (long)size2 * (long)size2;
    DIAG_EPS = 1e-14;

    jprev = (int*)malloc(size2*sizeof(int));
    jnext = (int*)malloc(size2*sizeof(int));
    kprev = (int*)malloc(size2*sizeof(int));
    knext = (int*)malloc(size2*sizeof(int));

    for (t = 0; t < size2; ++t) {
        kprev[t] = (t - 1 + size2) % size2;
        knext[t] = (t + 1) % size2;
        jprev[t] = kprev[t] * size2;
        jnext[t] = knext[t] * size2;
    }

    for (iter = 0; iter < iters; ++iter) {
        mpi_grid_exchange_bot_top(eps_x, size1, size2);
        mpi_grid_exchange_bot_top(eps_y, size1, size2);
        mpi_grid_exchange_bot_top(eps_z, size1, size2);

        /* RED */
        mpi_grid_exchange_bot_top(out, size1, size2);
        #pragma omp parallel for private(i,j,k,d,i0,i1,i2,j0,j1,j2,k1,k2,idx0,idx_x,idx_y,idx_z,ex_p,ex_m,ey_p,ey_m,ez_p,ez_m,diag,rhs)
        for (i = 0; i < size1; ++i) {
            i0 = (long)i * n2; i1 = i0 + n2; i2 = i0 - n2; d = (n_start + i) % 2;
            for (j = 0; j < size2; ++j) {
                j0 = (long)j * size2; j1 = jnext[j]; j2 = jprev[j];
                k = (2 - ((d + (j % 2)) % 2)) % 2;
                for (; k < size2; k += 2) {
                    k1 = knext[k]; k2 = kprev[k];
                    idx0 = i0 + j0 + k; idx_x = i2 + j0 + k; idx_y = i0 + j2 + k; idx_z = i0 + j0 + k2;

                    ex_p = eps_x[idx0]; ex_m = eps_x[idx_x];
                    ey_p = eps_y[idx0]; ey_m = eps_y[idx_y];
                    ez_p = eps_z[idx0]; ez_m = eps_z[idx_z];

                    diag = ex_p + ex_m + ey_p + ey_m + ez_p + ez_m + k2_screen[idx0];
                    if (diag < DIAG_EPS) diag = DIAG_EPS;

                    rhs = ex_p * out[i1 + j0 + k] +
                          ex_m * out[idx_x]       +
                          ey_p * out[i0 + j1 + k] +
                          ey_m * out[idx_y]       +
                          ez_p * out[i0 + j0 + k1]+
                          ez_m * out[idx_z]       -
                          in[idx0];

                    /* ——— SCRITTURA INTERMEDIA COME PRIMA ——— */
                    out[idx0] = rhs / diag;  /* mantiene il comportamento “di prima” */
                }
            }
        }

        /* BLACK */
        mpi_grid_exchange_bot_top(out, size1, size2);
        #pragma omp parallel for private(i,j,k,d,i0,i1,i2,j0,j1,j2,k1,k2,idx0,idx_x,idx_y,idx_z,ex_p,ex_m,ey_p,ey_m,ez_p,ez_m,diag,rhs)
        for (i = 0; i < size1; ++i) {
            i0 = (long)i * n2; i1 = i0 + n2; i2 = i0 - n2; d = (n_start + i) % 2;
            for (j = 0; j < size2; ++j) {
                j0 = (long)j * size2; j1 = jnext[j]; j2 = jprev[j];
                k = (1 - ((d + (j % 2)) % 2)) % 2;
                for (; k < size2; k += 2) {
                    k1 = knext[k]; k2 = kprev[k];
                    idx0 = i0 + j0 + k; idx_x = i2 + j0 + k; idx_y = i0 + j2 + k; idx_z = i0 + j0 + k2;

                    ex_p = eps_x[idx0]; ex_m = eps_x[idx_x];
                    ey_p = eps_y[idx0]; ey_m = eps_y[idx_y];
                    ez_p = eps_z[idx0]; ez_m = eps_z[idx_z];

                    diag = ex_p + ex_m + ey_p + ey_m + ez_p + ez_m + k2_screen[idx0];
                    if (diag < DIAG_EPS) diag = DIAG_EPS;

                    rhs = ex_p * out[i1 + j0 + k] +
                          ex_m * out[idx_x]       +
                          ey_p * out[i0 + j1 + k] +
                          ey_m * out[idx_y]       +
                          ez_p * out[i0 + j0 + k1]+
                          ez_m * out[idx_z]       -
                          in[idx0];

                    out[idx0] = rhs / diag;  /* idem */
                }
            }
        }
    }

    free(jprev); free(jnext); free(kprev); free(knext);
}

int multigrid_pb_apply(double *in, double *out, int s1, int s2, int n_start1, int sm, double *eps_x, double *eps_y, double *eps_z, double *k2_screen) {
    multigrid_pb_apply_recursive(in, out, s1, s2, n_start1, sm, eps_x, eps_y, eps_z, k2_screen);
}

void smooth_pb(double *in, double *out, int s1, int s2, double tol, double *eps_x, double *eps_y, double *eps_z, double *k2_screen) {
    // smooth_pb_jacobi(in, out, s1, s2, tol);
    smooth_pb_rbgs(in, out, s1, s2, tol, eps_x, eps_y, eps_z, k2_screen);
}
