#include <math.h>
#include <string.h>

#include "sphere_intersect.h"

/* ------------------------------------------------------------------ */
/* Helper    // Vincenzo Di Florio, 04.2026                                                           */
/* ------------------------------------------------------------------ */

static inline double min_image_1d(double d, double L)
{
    if (L <= 0.0) return d;
    return d - L * nearbyint(d / L);
}

/*
 * Solve the quadratic equation for the intersection between a parametric ray
 * P(t) = start + t * (edge_dir * h) and a sphere with center c and radius r.
 *
 * dx, dy, dz: components of (start - c), already minimum-image adjusted.
 * edge_dir: 0=X, 1=Y, 2=Z.
 * h: edge length.
 *
 * Writes unclamped values to t1_out and t2_out, with t1 <= t2.
 * Returns 1 if an intersection exists, 0 if the discriminant is negative.
 */
static int solve_sphere_edge(double dx, double dy, double dz,
                              double r, double h, int edge_dir,
                              double *t1_out, double *t2_out)
{
    /* A t^2 + B t + C = 0
     * A = h^2, always positive
     * B = 2 * d_along * h, where d_along is the component of (start-c) along the edge
     * C = |start - c|^2 - r^2
     */
    double d_along = (edge_dir == 0) ? dx : (edge_dir == 1) ? dy : dz;
    double A = h * h;
    double B = 2.0 * d_along * h;
    double C = dx*dx + dy*dy + dz*dz - r*r;

    double disc = B*B - 4.0*A*C;
    if (disc < 0.0) return 0;

    double sqrtd = sqrt(disc);
    double inv2A = 0.5 / A;
    *t1_out = (-B - sqrtd) * inv2A;
    *t2_out = (-B + sqrtd) * inv2A;
    return 1;
}

/* ------------------------------------------------------------------ */
/* is_in_molecule_sphere                                                */
/* ------------------------------------------------------------------ */

int is_in_molecule_sphere(const particles *p, double x, double y, double z, double L)
{
    for (int ip = 0; ip < p->n_p; ip++) {
        double dx = min_image_1d(x - p->pos[ip * 3 + 0], L);
        double dy = min_image_1d(y - p->pos[ip * 3 + 1], L);
        double dz = min_image_1d(z - p->pos[ip * 3 + 2], L);
        double r  = p->solv_radii[ip];
        if (dx*dx + dy*dy + dz*dz <= r*r) return 1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* sphere_edge_fraction                                                 */
/* ------------------------------------------------------------------ */

/* Maximum interval capacity, one per atom. Increase if N_atoms > 128. */
#define MAX_SPHERE_ATOMS 256

double sphere_edge_fraction(const particles *p,
    double x0, double y0, double z0,
    double h, int dir, double L)
{
    double lo[MAX_SPHERE_ATOMS], hi[MAX_SPHERE_ATOMS];
    int n_intervals = 0;

    for (int ip = 0; ip < p->n_p && n_intervals < MAX_SPHERE_ATOMS; ip++) {
        double dx = min_image_1d(x0 - p->pos[ip * 3 + 0], L);
        double dy = min_image_1d(y0 - p->pos[ip * 3 + 1], L);
        double dz = min_image_1d(z0 - p->pos[ip * 3 + 2], L);
        double r  = p->solv_radii[ip];

        double t1, t2;
        if (!solve_sphere_edge(dx, dy, dz, r, h, dir, &t1, &t2)) continue;

        /* Clamp to [0,1]. */
        if (t1 < 0.0) t1 = 0.0;
        if (t2 > 1.0) t2 = 1.0;
        if (t1 >= t2) continue;

        lo[n_intervals] = t1;
        hi[n_intervals] = t2;
        n_intervals++;
    }

    if (n_intervals == 0) return 0.0;
    if (n_intervals == 1) return hi[0] - lo[0];

    /* Sort by lo using insertion sort; n is small. */
    for (int i = 1; i < n_intervals; i++) {
        double lv = lo[i], hv = hi[i];
        int j = i - 1;
        while (j >= 0 && lo[j] > lv) {
            lo[j+1] = lo[j];
            hi[j+1] = hi[j];
            j--;
        }
        lo[j+1] = lv;
        hi[j+1] = hv;
    }

    /* Merge intervals and sum their lengths. */
    double frac    = 0.0;
    double cur_lo  = lo[0], cur_hi = hi[0];
    for (int i = 1; i < n_intervals; i++) {
        if (lo[i] <= cur_hi) {
            if (hi[i] > cur_hi) cur_hi = hi[i];
        } else {
            frac  += cur_hi - cur_lo;
            cur_lo = lo[i];
            cur_hi = hi[i];
        }
    }
    frac += cur_hi - cur_lo;

    if (frac < 0.0) frac = 0.0;
    if (frac > 1.0) frac = 1.0;
    return frac;
}

/* ------------------------------------------------------------------ */
/* sphere_edge_surface_inters                                           */
/* ------------------------------------------------------------------ */

/* Event: entering (+1) or leaving (-1) a sphere along the edge. */
typedef struct { double t; int atom; int sign; } SphereEvent;

#define MAX_EVENTS (2 * MAX_SPHERE_ATOMS)

int sphere_edge_surface_inters(
    const particles *p,
    double x0, double y0, double z0,
    double h, int dir, double L,
    double *t_out, double *nx_out, double *ny_out, double *nz_out,
    int max_inters)
{
    SphereEvent events[MAX_EVENTS];
    int n_events     = 0;
    int initial_count = 0; /* Number of spheres containing the start point (t=0). */

    for (int ip = 0; ip < p->n_p && n_events + 2 <= MAX_EVENTS; ip++) {
        double dx = min_image_1d(x0 - p->pos[ip * 3 + 0], L);
        double dy = min_image_1d(y0 - p->pos[ip * 3 + 1], L);
        double dz = min_image_1d(z0 - p->pos[ip * 3 + 2], L);
        double r  = p->solv_radii[ip];

        double t1, t2;
        if (!solve_sphere_edge(dx, dy, dz, r, h, dir, &t1, &t2)) continue;

        /* Sphere is beyond the edge end or before the edge start. */
        if (t2 <= 0.0) continue;
        if (t1 >= 1.0) continue;

        int starts_inside = (t1 < 0.0);
        int ends_inside   = (t2 > 1.0);

        if (starts_inside) {
            /* The edge starts inside this sphere. */
            initial_count++;
            if (!ends_inside) {
                /* Leaves the sphere at t2, inside [0,1]. */
                events[n_events].t    = t2;
                events[n_events].atom = ip;
                events[n_events].sign = -1;
                n_events++;
            }
        } else {
            /* Enters the sphere at t1, inside [0,1]. */
            events[n_events].t    = t1;
            events[n_events].atom = ip;
            events[n_events].sign = +1;
            n_events++;
            if (!ends_inside) {
                /* Leaves the sphere at t2, inside [0,1]. */
                events[n_events].t    = t2;
                events[n_events].atom = ip;
                events[n_events].sign = -1;
                n_events++;
            }
        }
    }

    if (n_events == 0) return 0;

    /* Sort events by t using insertion sort. */
    for (int i = 1; i < n_events; i++) {
        SphereEvent ev = events[i];
        int j = i - 1;
        while (j >= 0 && events[j].t > ev.t) {
            events[j+1] = events[j];
            j--;
        }
        events[j+1] = ev;
    }

    /* Sweep to find 0 <-> positive transitions, i.e. the external surface. */
    int count   = initial_count;
    int n_found = 0;

    for (int e = 0; e < n_events && n_found < max_inters; e++) {
        int prev_count = count;
        count += events[e].sign;

        /* External surface: transition between inside (count>0) and outside (count==0). */
        if ((prev_count == 0) != (count == 0)) {
            int    ip = events[e].atom;
            double t  = events[e].t;

            /* 3D position of the intersection point. */
            double px = x0, py = y0, pz = z0;
            if      (dir == 0) px += t * h;
            else if (dir == 1) py += t * h;
            else               pz += t * h;

            /* Outward sphere normal: unit vector (P - center) / r. */
            double cx = p->pos[ip * 3 + 0];
            double cy = p->pos[ip * 3 + 1];
            double cz = p->pos[ip * 3 + 2];
            double r  = p->solv_radii[ip];

            double npx = min_image_1d(px - cx, L);
            double npy = min_image_1d(py - cy, L);
            double npz = min_image_1d(pz - cz, L);
            double inv_r = 1.0 / r;

            t_out [n_found] = t;
            nx_out[n_found] = npx * inv_r;
            ny_out[n_found] = npy * inv_r;
            nz_out[n_found] = npz * inv_r;
            n_found++;
        }
    }

    return n_found;
}

#undef MAX_EVENTS
#undef MAX_SPHERE_ATOMS
