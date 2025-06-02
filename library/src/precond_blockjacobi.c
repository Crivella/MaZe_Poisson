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
    // vec_copy(A, tmp, (2 * B3) * (2 * B3));

    dgetri(A, B3);

    double *C = (double *)malloc((reps * B3) * (B3) * sizeof(double));

    // // Compute the product of A and its inverse to verify correctness   
    dgemm(tmp, A, C, reps*B3, B3, B3);

    // // Check if the product is close to the identity matrix
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

    // double *tmp = (double *)malloc(n3 * sizeof(double));

    // Treat the in vector as a n_blocks x BLOCK_SIZE^3 matrix
    // mpi_printf("Computing the product of the input vector with the inverse block matrix A\n");
    dgemm(in, A, out, n_blocks, b3, b3);

    // vec_copy(tmp, out, n3);

    // free(tmp);
}
