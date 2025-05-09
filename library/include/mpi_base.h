#ifndef __MYMPI_H
#define __MYMPI_H

#ifdef __MPI
#include <mpi.h>
#else
typedef int MPI_Comm;
#endif

typedef struct mpi_data {
    MPI_Comm comm;
    int rank;
    int size;

    int n_start;
    int n_loc;

    int *n_loc_list;
    int *n_start_list;

    int next_rank;
    int prev_rank;
} mpi_data;

int init_mpi();
void cleanup_mpi();

mpi_data *get_mpi_data();
int get_size();
int get_rank();
int get_n_loc();
int get_n_start();

void bcast_double(double *buffer, long int size, int root);
void allreduce_sum(double *buffer, long int size);

double * mpi_grid_allocate(int size1, int size2);
void mpi_grid_free(double *data, int n);
void mpi_grid_exchange_bot_top(double *grid, int size1, int size2);
void mpi_grid_collect_buffer(double *data, double *recv, int n);

#endif
