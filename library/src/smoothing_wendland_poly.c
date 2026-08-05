#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mpi_base.h"
#include "mp_structs.h"
#include "smoothing_wendland_poly.h"

typedef struct {
    int degree;
    double *coefficients;
    double *b1;
    double *b2;
    double *tmp;
} wendland_poly_kernel;

static void apply_mehrstellen_symbol(const double *in, double *out, int n) {
    const long n2 = (long)n * n;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const int im = i == 0 ? n - 1 : i - 1;
            const int ip = i + 1 == n ? 0 : i + 1;
            const int jm = j == 0 ? n - 1 : j - 1;
            const int jp = j + 1 == n ? 0 : j + 1;
            for (int k = 0; k < n; ++k) {
                const int km = k == 0 ? n - 1 : k - 1;
                const int kp = k + 1 == n ? 0 : k + 1;
                const long p = (long)i * n2 + (long)j * n + k;
                const double faces =
                    in[(long)im * n2 + (long)j * n + k] +
                    in[(long)ip * n2 + (long)j * n + k] +
                    in[(long)i * n2 + (long)jm * n + k] +
                    in[(long)i * n2 + (long)jp * n + k] +
                    in[(long)i * n2 + (long)j * n + km] +
                    in[(long)i * n2 + (long)j * n + kp];
                const double edges =
                    in[(long)im * n2 + (long)jm * n + k] +
                    in[(long)im * n2 + (long)jp * n + k] +
                    in[(long)ip * n2 + (long)jm * n + k] +
                    in[(long)ip * n2 + (long)jp * n + k] +
                    in[(long)im * n2 + (long)j * n + km] +
                    in[(long)im * n2 + (long)j * n + kp] +
                    in[(long)ip * n2 + (long)j * n + km] +
                    in[(long)ip * n2 + (long)j * n + kp] +
                    in[(long)i * n2 + (long)jm * n + km] +
                    in[(long)i * n2 + (long)jm * n + kp] +
                    in[(long)i * n2 + (long)jp * n + km] +
                    in[(long)i * n2 + (long)jp * n + kp];
                /* Z = 1 + 3 L_M / 8 maps the Mehrstellen spectrum to [-1, 1]. */
                out[p] = -0.5 * in[p] + 0.125 * faces + 0.0625 * edges;
            }
        }
    }
}

void smooth_charges_wendland_poly_init(grid *grid) {
    if (get_size() != 1 || grid->n_local != grid->n) {
        mpi_fprintf(stderr, "Polynomial Wendland smoothing currently requires one MPI rank\n");
        exit(EXIT_FAILURE);
    }

    const long size = (long)grid->n * grid->n * grid->n;
    wendland_poly_kernel *kernel = calloc(1, sizeof(*kernel));
    if (kernel != NULL) {
        kernel->degree = -1;
        kernel->b1 = malloc(size * sizeof(*kernel->b1));
        kernel->b2 = malloc(size * sizeof(*kernel->b2));
        kernel->tmp = malloc(size * sizeof(*kernel->tmp));
    }
    if (kernel == NULL || kernel->b1 == NULL || kernel->b2 == NULL ||
        kernel->tmp == NULL) {
        mpi_fprintf(stderr, "Unable to allocate polynomial Wendland buffers\n");
        exit(EXIT_FAILURE);
    }
    grid->smoothing_kernel = kernel;
}

void smooth_charges_wendland_poly_set_coefficients(
    grid *grid, int degree, const double *coefficients
) {
    wendland_poly_kernel *kernel = grid->smoothing_kernel;
    if (kernel == NULL || degree < 1 || coefficients == NULL) {
        mpi_fprintf(stderr, "Invalid polynomial Wendland coefficients\n");
        exit(EXIT_FAILURE);
    }
    double *copy = malloc((degree + 1) * sizeof(*copy));
    if (copy == NULL) {
        mpi_fprintf(stderr, "Unable to allocate polynomial Wendland coefficients\n");
        exit(EXIT_FAILURE);
    }
    memcpy(copy, coefficients, (degree + 1) * sizeof(*copy));
    free(kernel->coefficients);
    kernel->coefficients = copy;
    kernel->degree = degree;
}

void smooth_charges_wendland_poly(grid *grid) {
    wendland_poly_kernel *kernel = grid->smoothing_kernel;
    if (kernel == NULL || kernel->coefficients == NULL || kernel->degree < 1) {
        mpi_fprintf(stderr, "Polynomial Wendland coefficients were not initialized\n");
        exit(EXIT_FAILURE);
    }
    const long size = (long)grid->n * grid->n * grid->n;
    const double *source = grid->q;
    double *b1 = kernel->b1;
    double *b2 = kernel->b2;
    double *tmp = kernel->tmp;
    memset(b1, 0, size * sizeof(*b1));
    memset(b2, 0, size * sizeof(*b2));

    for (int degree = kernel->degree; degree >= 1; --degree) {
        apply_mehrstellen_symbol(b1, tmp, grid->n);
        #pragma omp parallel for schedule(static)
        for (long p = 0; p < size; ++p) {
            tmp[p] = 2.0 * tmp[p] - b2[p]
                   + kernel->coefficients[degree] * source[p];
        }
        double *old_b2 = b2;
        b2 = b1;
        b1 = tmp;
        tmp = old_b2;
    }
    apply_mehrstellen_symbol(b1, tmp, grid->n);
    #pragma omp parallel for schedule(static)
    for (long p = 0; p < size; ++p) {
        grid->q[p] = tmp[p] - b2[p]
                   + kernel->coefficients[0] * source[p];
    }
}

void smooth_charges_wendland_poly_free(grid *grid) {
    wendland_poly_kernel *kernel = grid->smoothing_kernel;
    if (kernel == NULL) {
        return;
    }
    free(kernel->b1);
    free(kernel->b2);
    free(kernel->tmp);
    free(kernel->coefficients);
    free(kernel);
    grid->smoothing_kernel = NULL;
}
