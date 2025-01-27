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

    long int buffer_size;
    double *bot;
    double *top;
} mpi_data;

int init_mpi();
int init_mpi_grid(int n);
void cleanup_mpi();

int get_size();
int get_rank();
int get_n_loc();
int get_n_start();
void exchange_bot_top(double *bot, double *top, double **bot_recv, double **top_recv);

void allreduce_sum(double *buffer, long int size);
void collect_grid_buffer(double *data, double *recv, int n);

#endif
