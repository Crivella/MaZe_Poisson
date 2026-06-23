#include <stdlib.h>
#include <string.h>

#include "mp_structs.h"
#include "mpi_base.h"


#ifdef __SCAFACOS

#include <fcs.h>

void scafacos_check_result(FCSResult result) {
    if (result != FCS_SUCCESS) {
        mpi_fprintf(
            stderr, "Error in ScaFaCoS: `%s` %s\n",
            fcs_result_get_function(result), fcs_result_get_message(result)
        );
        exit(1);
    }
}

void scafacos_grid_init(grid * grid) {
    FCS handle;
    FCSResult result;

    mpi_data *mpid = get_mpi_data();
    scafacos_check_result(fcs_init(&handle, "fmm", mpid->comm));

    grid->fcs_handle = handle;
    mpi_printf("ScaFaCoS initialized successfully.\n");

    fcs_float box_a[3] = {grid->L, 0, 0};
    fcs_float box_b[3] = {0, grid->L, 0};
    fcs_float box_c[3] = {0, 0, grid->L};
    fcs_int periodicity[3] = {1, 1, 1};

    scafacos_check_result(fcs_set_box_a(handle, box_a));
    scafacos_check_result(fcs_set_box_b(handle, box_b));
    scafacos_check_result(fcs_set_box_c(handle, box_c));

    scafacos_check_result(fcs_set_periodicity(handle, periodicity));

    grid->tuned = 0;

    grid->phi_p = NULL;
    grid->phi_n = NULL;
    grid->fcs_potential = NULL;

    grid->init_field = scafacos_grid_init_field;
    grid->update_field = scafacos_grid_update_field;
    grid->update_charges = scafacos_grid_update_charges;
}

void scafacos_grid_cleanup(grid * grid) {
    if (grid->fcs_pos != NULL) {
        free(grid->fcs_pos);
    }
    if (grid->fcs_charges != NULL) {
        free(grid->fcs_charges);
    }
    if (grid->fcs_potential != NULL) {
        free(grid->fcs_potential);
    }
    if (grid->phi_p != NULL) {
        free(grid->phi_p);
    }
    if (grid->phi_n != NULL) {
        free(grid->phi_n);
    }
    
    fcs_destroy(grid->fcs_handle);
}

void scafacos_grid_init_field(grid *grid) {
    FCS handle = (FCS)grid->fcs_handle;

    scafacos_check_result(fcs_run(
        handle,
        grid->n_p,
        grid->fcs_pos,
        grid->fcs_charges,
        grid->phi_n,
        grid->fcs_potential
    ));
}

int scafacos_grid_update_field(grid *grid) {
    memcpy(grid->phi_p, grid->phi_n, 3 * grid->n_p * sizeof(double));
    scafacos_grid_init_field(grid);
}   

void scafacos_tune(grid *grid, particles *p) {
    FCS handle = (FCS)grid->fcs_handle;

    int np_local = p->np_local;
    int np_start = p->np_start;

    grid->n_p = np_local;

    grid->fcs_pos = (double *)malloc(3 * np_local * sizeof(double));
    grid->fcs_charges = (double *)malloc(np_local * sizeof(double));
    grid->fcs_potential = (double *)malloc(np_local * sizeof(double));
    grid->phi_n = (double *)malloc(3 * np_local * sizeof(double));
    grid->phi_p = (double *)malloc(3 * np_local * sizeof(double));

    for (int i=0; i<np_local; i++) {
        grid->fcs_charges[i] = p->charges[np_start + i];
    }

    scafacos_check_result(fcs_set_total_particles(handle, p->n_p));
    scafacos_check_result(fcs_fmm_set_internal_tuning(handle, FCS_FMM_HOMOGENOUS_SYSTEM));
    scafacos_check_result(fcs_tune(handle, p->n_p, p->pos, p->charges));

    grid->tuned = 1;
}

double scafacos_grid_update_charges(grid *grid, particles *p) {
    if (! grid->tuned) {
        scafacos_tune(grid, p);
    }

    int i_loc;
    int np_local = p->np_local;
    int np_start = p->np_start;
    for (int i=0; i<np_local; i++) {
        i_loc = np_start + i;
        grid->fcs_pos[3*i + 0] = p->pos[3*i_loc + 0];
        grid->fcs_pos[3*i + 1] = p->pos[3*i_loc + 1];
        grid->fcs_pos[3*i + 2] = p->pos[3*i_loc + 2];
    }

    return 0.0;
}

#else  // __SCAFACOS

void scafacos_grid_init(grid * grid) {
    mpi_fprintf(stderr, "Error: ScaFaCoS is not enabled. Please compile with ScaFaCoS support.\n");
    exit(1);
}

void scafacos_grid_cleanup(grid * grid) {
    mpi_fprintf(stderr, "Error: ScaFaCoS is not enabled. Please compile with ScaFaCoS support.\n");
    exit(1);
}

void scafacos_grid_init_field(grid *grid) {
    mpi_fprintf(stderr, "Error: ScaFaCoS is not enabled. Please compile with ScaFaCoS support.\n");
    exit(1);
}

int scafacos_grid_update_field(grid *grid) {
    mpi_fprintf(stderr, "Error: ScaFaCoS is not enabled. Please compile with ScaFaCoS support.\n");
    exit(1);
}

#endif  // __SCAFACOS
