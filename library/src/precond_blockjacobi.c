#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "linalg.h"
#include "mpi_base.h"
#include "mp_structs.h"

#define BLOCK_SIZE 10

double *A = NULL;

void precond_blockjacobi_init() {
    if (A != NULL) {
        return;
    }

    int B = BLOCK_SIZE;
    long int B2 = B * B;
    long int B3 = B2 * B;

    int Bm1 = B - 1;

    A = (double *)calloc(B3 * B3, sizeof(double));

    long int idx;
    long int i0, i1, i2;
    long int j0, j1, j2;
    long int k0, k1, k2;
    for (int i = 0; i < B; i++) {
        i0 = i * B2;
        i1 = i0 + B2;
        i2 = i0 - B2;
        for (int j = 0; j < B; j++) {
            j0 = j * B;
            j1 = j0 + B;
            j2 = j0 - B;
            for (int k = 0; k < B; k++) {
                k0 = k;
                k1 = k0 + 1;
                k2 = k0 - 1;

                idx = B3 * (i0 + j0 + k0);
                A[idx + i0 + j0 + k0] = -6.0;  // Diagonal element
                if (i > 0) {
                    A[idx + i2 + j0 + k0] = 1.0;  // Left neighbor
                }
                if (i < Bm1) {
                    A[idx + i1 + j0 + k0] = 1.0;  // Right neighbor
                }
                if (j > 0) {
                    A[idx + i0 + j2 + k0] = 1.0;  // Bottom neighbor
                }
                if (j < Bm1) {
                    A[idx + i0 + j1 + k0] = 1.0;  // Top neighbor
                }
                if (k > 0) {
                    A[idx + i0 + j0 + k2] = 1.0;  // Back neighbor
                }
                if (k < Bm1) {
                    A[idx + i0 + j0 + k1] = 1.0;  // Front neighbor
                }
            }
        }
    }

    dgetri(A, B3);
}

void precond_blockjacobi_cleanup() {
    if (A != NULL) {
        free(A);
        A = NULL;
    }
}

// Reorder the grid input vector to match the block structure
void reorder_in(double *in, double *out, int s1, int n) {
    long int n3 = s1 * n * n;
    int b = BLOCK_SIZE;
    long int b2 = BLOCK_SIZE * BLOCK_SIZE;
    long int b3 = BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE;

    long int n2 = n * n;

    long int x,y,z;
    long int idx_in, idx_out;
    long int i1, j1, k1;

    if (s1 % BLOCK_SIZE != 0) {
        mpi_fprintf(stderr, "Error: The size of the input vector is not a multiple of the block size (%d)\n", BLOCK_SIZE);
        mpi_fprintf(stderr, "Padding is not implemented yet\n");
        exit(1);
    }

    int block_1d = s1 / BLOCK_SIZE;
    long int b1d_2 = block_1d * block_1d;
    long int block_num;

    #pragma omp parallel for private(x, y, z, idx_in, idx_out, i1, j1, k1, block_num)
    // Loop over the blocks in the 3D grid
    for (int i = 0; i <  block_1d; i++) {
        i1 = i * BLOCK_SIZE;
        for (int j = 0; j < block_1d; j++) {
            j1 = j * BLOCK_SIZE;
            // printf("i = %d, j = %d\n", i, j);
            for (int k = 0; k < block_1d; k++) {
                k1 = k * BLOCK_SIZE;
                block_num = i * b1d_2 + j * block_1d + k;
                // Loop over the elements in the block
                for (int ii = 0; ii < BLOCK_SIZE; ii++) {
                    x = i1 + ii;
                    for (int jj = 0; jj < BLOCK_SIZE; jj++) {
                        y = j1 + jj;
                        for (int kk = 0; kk < BLOCK_SIZE; kk++) {
                            z = k1 + kk;
                            idx_in = x * n2 + y * n + z;
                            idx_out = block_num * b3 + ii * b2 + jj * b + kk;
                            out[idx_out] = in[idx_in];
                        }
                    }
                }
            }
        }
    }
}

// Reorder the output vector to match the original grid structure
void reorder_out(double *in, double *out, int s1, int n) {
    long int n3 = s1 * n * n;
    int b = BLOCK_SIZE;
    long int b2 = BLOCK_SIZE * BLOCK_SIZE;
    long int b3 = BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE;

    long int n2 = n * n;

    long int x,y,z;
    long int idx_in, idx_out;
    long int i1, j1, k1;

    int block_1d = s1 / BLOCK_SIZE;
    long int b1d_2 = block_1d * block_1d;
    long int block_num;

    #pragma omp parallel for private(x, y, z, idx_in, idx_out, i1, j1, k1, block_num)
    // Loop over the blocks in the 3D grid
    for (int i = 0; i <  block_1d; i++) {
        i1 = i * BLOCK_SIZE;
        for (int j = 0; j < block_1d; j++) {
            j1 = j * BLOCK_SIZE;
            for (int k = 0; k < block_1d; k++) {
                k1 = k * BLOCK_SIZE;
                block_num = i * b1d_2 + j * block_1d + k;
                // Loop over the elements in the block
                for (int ii = 0; ii < BLOCK_SIZE; ii++) {
                    x = i1 + ii;
                    for (int jj = 0; jj < BLOCK_SIZE; jj++) {
                        y = j1 + jj;
                        for (int kk = 0; kk < BLOCK_SIZE; kk++) {
                            z = k1 + kk;
                            idx_in = x * n2 + y * n + z;
                            idx_out = block_num * b3 + ii * b2 + jj * b + kk;
                            out[idx_in] = in[idx_out];
                        }
                    }
                }
            }
        }
    }
}

void precond_blockjacobi_apply(double *in, double *out, int s1, int s2, int n_start) {
    long int n3 = s1 * s2 * s2;

    long int b3 = BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE;

    int mpi_size = get_size();

    if (mpi_size > 1) {
        mpi_fprintf(stderr, "Error: Block Jacobi preconditioner is not implemented for MPI yet\n");
        exit(1);
    }

    if (n3 % b3 != 0) {
        mpi_fprintf(stderr, "Error: The size of the input vector is not a multiple of the block size (%ld)\n", b3);
        mpi_fprintf(stderr, "Padding is not implemented yet\n");
        exit(1);
    }
    
    long int n_blocks = n3 / b3;

    double *tmp1 = (double *)malloc(n3 * sizeof(double));
    double *tmp2 = (double *)malloc(n3 * sizeof(double));

    reorder_in(in, tmp1, s1, s2);

    // Treat the in vector as a n_blocks x BLOCK_SIZE^3 matrix
    // mpi_printf("Computing the product of the input vector with the inverse block matrix A\n");
    dgemm(tmp1, A, tmp2, n_blocks, b3, b3);

    reorder_out(tmp2, out, s1, s2);

    free(tmp1);
    free(tmp2);
}
