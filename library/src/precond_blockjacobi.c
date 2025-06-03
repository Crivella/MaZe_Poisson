#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "linalg.h"
#include "mpi_base.h"
#include "mp_structs.h"

#define BLOCK_SIZE 3

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

    for (long int i = 0; i < B3; i++) {
        for (long int j = 0; j < B3; j++) {
            printf("%3.0f ", i, j, A[B3 * i + j]);
        }
        printf("\n");
    }
    printf("-------------------------------------------------------------------------------\n");

    // Compute the inverse of the block matrix A
    // mpi_printf("Computing the inverse of the block matrix A\n");
    // Test dgemm against a 2x2 block copy of A
    int reps = 4;
    double *tmp = (double *)calloc((reps * B3) * (B3), sizeof(double));
    for (long int i = 0; i < B3; i++) {
        for (long int j = 0; j < B3; j++) {
            for (int r=0; r < reps; r++) {
                // Copy the block matrix A into a larger matrix tmp
                // This is just for testing purposes, to see if dgemm works correctly
                tmp[(B3) * (i + r*B3) + j] = A[B3 * i + j];
            }
        }
    }

    dgetri(A, B3);

    double *C = (double *)malloc((reps * B3) * (B3) * sizeof(double));

    // Compute the product of A and its inverse to verify correctness   
    dgemm(tmp, A, C, reps*B3, B3, B3);

    // Check if the product is close to the identity matrix
    for (long int i = 0; i < reps*B3; i++) {
        for (long int j = 0; j < B3; j++) {
            double v = C[B3 * i + j];
            if (fabs(v) < 1e-9) {
                v = 0;
            } 
            printf("%3.0f ", i, j, v);
        }
        printf("\n");
    }
    printf("-------------------------------------------------------------------------------\n");

    for (long int i = 0; i < B3; i++) {
        for (long int j = 0; j < B3; j++) {
            printf("%7.4f ", i, j, A[B3 * i + j]);
        }
        printf("\n");
    }    
    printf("-------------------------------------------------------------------------------\n");
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

    // int *tmp1 = (int *)calloc(n3, sizeof(int));
    // int *tmp2 = (int *)calloc(n3, sizeof(int));
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
                            // if (tmp2[idx_in] == 1) {
                            //     mpi_printf("Error: reading twice from index %ld in input vector\n", idx_in);
                            //     mpi_printf("block_1d = %d, BLOCK_SIZE = %d, s1 = %d, n = %d\n", block_1d, BLOCK_SIZE, s1, n);
                            //     mpi_printf("i = %d, j = %d, k = %d, ii = %d, jj = %d, kk = %d\n", i, j, k, ii, jj, kk);
                            //     mpi_printf("x = %ld, y = %ld, z = %ld\n", x, y, z);
                            //     exit(1);
                            // }
                            // tmp1[idx_out] = 1;
                            // tmp2[idx_in] = 1;
                        }
                    }
                }
            }
        }
    }

    // int cnt = 0;
    // for (long int i = 0; i < n3; i++) {
    //     if (tmp1[i] != 1) {
    //         mpi_printf("Error: index %ld on idx_out was not mapped correctly\n", i);
    //         cnt += 1;
    //     }
    //     if (tmp2[i] != 1) {
    //         mpi_printf("Error: index %ld on idx_in was not mapped correctly\n", i);
    //         cnt += 1;
    //     }
    //     if (cnt > 300) {
    //         mpi_printf("Too many errors, exiting...\n");
    //         mpi_printf("block_1d = %d, BLOCK_SIZE = %d, s1 = %d, n = %d\n", block_1d, BLOCK_SIZE, s1, n);
    //         exit(1);
    //     }
    // }
    // free(tmp1);
    // free(tmp2);
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

    if (n3 % b3 != 0) {
        mpi_fprintf(stderr, "Error: The size of the input vector is not a multiple of the block size (%ld)\n", b3);
        mpi_fprintf(stderr, "Padding is not implemented yet\n");
        exit(1);
    }


    // mpi_printf("Applying block Jacobi preconditioner with block size %d\n", BLOCK_SIZE);
    // mpi_printf("To grid vector of size %d * %d * %d = %ld\n", s1, s2, s2, n3);
    
    long int n_blocks = n3 / b3;

    // mpi_printf("Copying `in` array into a temporary array just because...");

    // double *in_tmp = (double *)malloc(n3 * sizeof(double));
    // vec_copy(in, in_tmp, n3);

    double *tmp1 = (double *)malloc(n3 * sizeof(double));
    double *tmp2 = (double *)malloc(n3 * sizeof(double));

    // for (long int i = 0; i < 5000; i++) {
    //     if (fabs(in[i]) > 1e-6) {
    //         printf("%d ", i);
    //     } 
    // }
    // printf("\n");

    reorder_in(in, tmp1, s1, s2);

    // Treat the in vector as a n_blocks x BLOCK_SIZE^3 matrix
    // mpi_printf("Computing the product of the input vector with the inverse block matrix A\n");
    dgemm(tmp1, A, tmp2, n_blocks, b3, b3);

    reorder_out(tmp2, out, s1, s2);

    // for (long int i = 0; i < 5000; i++) {
    //     if (fabs(out[i]) > 1e-6) {
    //         printf("%d ", i);
    //     } 
    // }
    // printf("\n");
    // exit(0);

    // vec_copy(in, out, n3);

    free(tmp1);
    free(tmp2);
}
