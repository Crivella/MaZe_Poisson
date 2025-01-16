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

    int next_rank;
    int prev_rank;
    int enabled;

    long int buffer_size;
    double *bot;
    double *top;
} mpi_data;

int init_mpi(int n, void *comm_ptr);
void cleanup_mpi();

int get_n_loc();
int get_n_start();
void exchange_bot_top(double *bot, double *top, double **bot_recv, double **top_recv);

void allreduce_double(double *value);
void allreduce_buffer(double *buffer, long int size);

#endif
