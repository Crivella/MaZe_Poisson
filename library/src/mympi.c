#include <stdio.h>
#include <stdlib.h>

#include "mympi.h"

mpi_data *global_mpi_data = NULL;

int get_n_loc() {
    return global_mpi_data->n_loc;
}

int get_n_start() {
    return global_mpi_data->n_start;
}

#ifdef __MPI

int init_mpi(int n, void *comm_ptr) {
    if (global_mpi_data != NULL) {
        return 0;
    }
    int rank, size, n_loc, n_start, next_rank, prev_rank;
    int div, mod;
    MPI_Comm comm = *((MPI_Comm *)comm_ptr);
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    div = n / size;
    mod = n % size;

    if (rank < mod) {
        n_loc = div + 1;
        n_start = rank * n_loc;
    } else {
        n_loc = div;
        n_start = rank * n_loc + mod;
    }

    next_rank = (rank + 1) % size;
    prev_rank = (rank - 1 + size) % size;

    global_mpi_data = (mpi_data *)malloc(sizeof(mpi_data));

    global_mpi_data->comm = comm;
    global_mpi_data->rank = rank;
    global_mpi_data->size = size;
    global_mpi_data->n_start = n_start;
    global_mpi_data->n_loc = n_loc;
    global_mpi_data->next_rank = next_rank;
    global_mpi_data->prev_rank = prev_rank;
    if (size == 1) {
        global_mpi_data->enabled = 0;
    } else {
        global_mpi_data->enabled = 1;
    }

    global_mpi_data->buffer_size = n * n;
    global_mpi_data->bot = (double *)malloc(n * n * sizeof(double));
    global_mpi_data->top = (double *)malloc(n * n * sizeof(double));
    printf("MPI initialized with %d processes\n", size);

    return global_mpi_data->enabled;
}

void cleanup_mpi() {
    printf("Cleaning up MPI rank %d\n", global_mpi_data->rank);
    if (global_mpi_data != NULL) {
        free(global_mpi_data->bot);
        free(global_mpi_data->top);
        free(global_mpi_data);
    }
}

void exchange_bot_top(double *bot, double *top, double **bot_recv, double **top_recv) {
    MPI_Sendrecv(
        top, global_mpi_data->buffer_size, MPI_DOUBLE, global_mpi_data->next_rank, 0,
        global_mpi_data->bot, global_mpi_data->buffer_size, MPI_DOUBLE, global_mpi_data->prev_rank, 0,
        global_mpi_data->comm, MPI_STATUS_IGNORE
    );
    MPI_Sendrecv(
        bot, global_mpi_data->buffer_size, MPI_DOUBLE, global_mpi_data->prev_rank, 0,
        global_mpi_data->top, global_mpi_data->buffer_size, MPI_DOUBLE, global_mpi_data->next_rank, 0,
        global_mpi_data->comm, MPI_STATUS_IGNORE
    );

    *bot_recv = global_mpi_data->bot;
    *top_recv = global_mpi_data->top;
}

void allreduce_double(double *value) {
    MPI_Allreduce(MPI_IN_PLACE, value, 1, MPI_DOUBLE, MPI_SUM, global_mpi_data->comm);
}

void allreduce_buffer(double *buffer, long int size) {
    MPI_Allreduce(
        MPI_IN_PLACE, buffer, size, MPI_DOUBLE, MPI_SUM, global_mpi_data->comm
    );
}

#else

int init_mpi(int n, void *comm_ptr) {
    global_mpi_data = (mpi_data *)malloc(sizeof(mpi_data));
    global_mpi_data->n_loc = n;
    global_mpi_data->enabled = 0;

    return 0;
}

void cleanup_mpi() {
    if (global_mpi_data != NULL) {
        free(global_mpi_data);
    }
}

void exchange_bot_top(double *bot, double *top, double **bot_recv, double **top_recv) {
    *bot_recv = bot;
    *top_recv = top;
}

void allreduce_double(double *value) {
    // Do nothing
}

void allreduce_buffer(double *buffer, long int size) {
    // Do nothing
}

#endif

