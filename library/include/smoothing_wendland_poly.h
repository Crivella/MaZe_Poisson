#ifndef SMOOTHING_WENDLAND_POLY_H
#define SMOOTHING_WENDLAND_POLY_H

struct grid;

void smooth_charges_wendland_poly_init(struct grid *grid);
void smooth_charges_wendland_poly_set_coefficients(
    struct grid *grid, int degree, const double *coefficients
);
void smooth_charges_wendland_poly(struct grid *grid);
void smooth_charges_wendland_poly_free(struct grid *grid);

#endif // SMOOTHING_WENDLAND_POLY_H
