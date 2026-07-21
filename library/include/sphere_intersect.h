#ifndef SPHERE_INTERSECT_H
#define SPHERE_INTERSECT_H

#include "mp_structs.h"
// Vincenzo Di Florio, 04.2026
/*
 * Analytic sphere-edge geometry helpers for the no_NS branch.
 * These replace NanoShaper for surface-grid intersection calculations.
 * The molecular surface is the union of VdW spheres with radii solv_radii.
 */

/*
 * Return 1 if point (x,y,z) is inside at least one atomic sphere.
 * L: box length, used for minimum-image PBC; 0 disables PBC.
 */
int is_in_molecule_sphere(const particles *p, double x, double y, double z, double L);

/*
 * Compute the fraction [0,1] of the edge inside the union of spheres.
 * Edge: from (x0,y0,z0) to (x0+h,y0,z0) for dir=0, analogously for dir=1,2.
 * Intervals are merged to handle overlapping spheres.
 */
double sphere_edge_fraction(const particles *p,
    double x0, double y0, double z0,
    double h, int dir, double L);

/*
 * Find intersections between the edge and the surface of the union of spheres.
 * These are the points where the edge transitions between inside and outside.
 * Writes parameter t in [0,1] to t_out[] and the outward normal to nx/ny/nz_out[].
 * Returns the number of intersections found, capped by max_inters.
 */
int sphere_edge_surface_inters(
    const particles *p,
    double x0, double y0, double z0,
    double h, int dir, double L,
    double *t_out, double *nx_out, double *ny_out, double *nz_out,
    int max_inters);

#endif /* SPHERE_INTERSECT_H */
