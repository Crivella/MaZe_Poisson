#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "mympi.h"

mpi_data *global_mpi_data = NULL;

int get_size() {
    return global_mpi_data->size;
}

int get_rank() {
    return global_mpi_data->rank;
}

int get_n_loc() {
    return global_mpi_data->n_loc;
}

int get_n_start() {
    return global_mpi_data->n_start;
}

#ifdef __MPI

int init_mpi() {
    if (global_mpi_data != NULL) {
        return 0;
    }
    int rank, size, next_rank, prev_rank;

    MPI_Init(NULL, NULL);

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    next_rank = (rank + 1) % size;
    prev_rank = (rank - 1 + size) % size;

    global_mpi_data = (mpi_data *)malloc(sizeof(mpi_data));

    global_mpi_data->comm = comm;
    global_mpi_data->rank = rank;
    global_mpi_data->size = size;
    global_mpi_data->next_rank = next_rank;
    global_mpi_data->prev_rank = prev_rank;

    global_mpi_data->bot = NULL;
    global_mpi_data->top = NULL;

    return global_mpi_data->size;
}

int init_mpi_grid(int n) {
    int rank, size, n_loc, n_start;
    double div, mod;

    long int buffer_size = n * n;

    rank = global_mpi_data->rank;
    size = global_mpi_data->size;

    div = n / size;
    mod = n % size;

    if (rank < mod) {
        n_loc = div + 1;
        n_start = rank * n_loc;
    } else {
        n_loc = div;
        n_start = rank * n_loc + mod;
    }

    global_mpi_data->n_start = n_start;
    global_mpi_data->n_loc = n_loc;
    global_mpi_data->buffer_size = buffer_size;
    if (global_mpi_data->size > 1) {
        global_mpi_data->bot = (double *)malloc(buffer_size * sizeof(double));
        global_mpi_data->top = (double *)malloc(buffer_size * sizeof(double));
    }

    return n_loc;
}

void cleanup_mpi() {
    if (global_mpi_data != NULL) {
        if (global_mpi_data->bot != NULL) {
            free(global_mpi_data->bot);
        }
        if (global_mpi_data->top != NULL) {
            free(global_mpi_data->top);
        }
        free(global_mpi_data);
    }
    MPI_Finalize();
}

void exchange_bot_top(double *bot, double *top, double **bot_recv, double **top_recv) {
    if (global_mpi_data->size == 1) {
        *bot_recv = top;
        *top_recv = bot;
    } else {
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
}

void allreduce_double(double *value) {
    if (global_mpi_data->size > 1) {
        MPI_Allreduce(MPI_IN_PLACE, value, 1, MPI_DOUBLE, MPI_SUM, global_mpi_data->comm);
    }
}

void allreduce_buffer(double *buffer, long int size) {
    if (global_mpi_data->size > 1) {
        MPI_Allreduce(MPI_IN_PLACE, buffer, size, MPI_DOUBLE, MPI_SUM, global_mpi_data->comm);
    }
}

void broadcast_int(long int *value) {
    if (global_mpi_data->size > 1) {
        MPI_Bcast(value, 1, MPI_LONG, 0, global_mpi_data->comm);
    }
}

double *broadcast_buffer(double *data, long int size) {
    double *result = data;
    if (global_mpi_data->size > 1) {
        broadcast_int(&size);
        if (global_mpi_data->rank > 0) {
            result = (double *)malloc(size * sizeof(double));
        }
        MPI_Bcast(result, size, MPI_DOUBLE, 0, global_mpi_data->comm);
    }
    return result;
}

void collect_grid_buffer(double *data, double *recv, int n) {
    int n_loc = global_mpi_data->n_loc;
    int size = global_mpi_data->size;
    int rank = global_mpi_data->rank;

    long int n2 = n * n;
    long int n3_loc = n_loc * n2;
    long int n3 = n * n2;

    if (size > 1) {
        if (rank == 0) {
            double *app = malloc(n3 * sizeof(double));
            memcpy(recv, data, n3_loc * sizeof(double));
            for (int i=1; i<size; i++) {
                int n_loc_start = 0;
                int n_loc = 0;
                MPI_Recv(&n_loc_start, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&n_loc, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv + n_loc_start * n2, n3_loc, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        } else {
            MPI_Send(&global_mpi_data->n_start, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(&global_mpi_data->n_loc, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(data, n3_loc, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }
}


#else

int init_mpi() {
    global_mpi_data = (mpi_data *)malloc(sizeof(mpi_data));

    return 1;
}

int init_mpi_grid(int n) {
    global_mpi_data->n_loc = n;
    global_mpi_data->n_start = 0;

    return n;
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

double *collect_grid_buffer(double *data, int n) {
    return data;
}

#endif

