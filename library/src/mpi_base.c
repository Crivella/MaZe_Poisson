#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "mpi_base.h"

mpi_data *global_mpi_data = NULL;

mpi_data *get_mpi_data() {
    return global_mpi_data;
}

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
        return global_mpi_data->size;
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

    global_mpi_data->n_loc_list = (int *)malloc(size * sizeof(int));
    global_mpi_data->n_start_list = (int *)malloc(size * sizeof(int));

    global_mpi_data->bot = NULL;
    global_mpi_data->top = NULL;

    return global_mpi_data->size;
}

void cleanup_mpi() {
    if (global_mpi_data != NULL) {
        if (global_mpi_data->bot != NULL) {
            free(global_mpi_data->bot);
        }
        if (global_mpi_data->top != NULL) {
            free(global_mpi_data->top);
        }
        if (global_mpi_data->n_loc_list != NULL) {
            free(global_mpi_data->n_loc_list);
        }
        if (global_mpi_data->n_start_list != NULL) {
            free(global_mpi_data->n_start_list);
        }
        free(global_mpi_data);
        global_mpi_data = NULL;
        MPI_Finalize();
    }
}

void exchange_bot_top(double *bot, double *top, double **bot_recv, double **top_recv) {
    // Skip loop communication if the processor is holding no data
    if (global_mpi_data->n_loc == 0) {
        return;
    }
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

void allreduce_sum(double *buffer, long int count) {
    if (global_mpi_data->size > 1) {
        MPI_Allreduce(MPI_IN_PLACE, buffer, count, MPI_DOUBLE, MPI_SUM, global_mpi_data->comm);
    }
}

void bcast_double(double *buffer, long int size, int root) {
    if (global_mpi_data->size > 1) {
        MPI_Bcast(buffer, size, MPI_DOUBLE, root, global_mpi_data->comm);
    }
}

void collect_grid_buffer(double *data, double *recv, int n) {
    int n_loc = global_mpi_data->n_loc;
    int n_loc_start;
    int size = global_mpi_data->size;
    int rank = global_mpi_data->rank;

    long int n2 = n * n;
    long int n3_loc = n_loc * n2;

    if (rank == 0) {
        memcpy(recv, data, n3_loc * sizeof(double));
        for (int i=1; i<size; i++) {
            n_loc = global_mpi_data->n_loc_list[i];
            n_loc_start = global_mpi_data->n_start_list[i];
            MPI_Recv(recv + n_loc_start * n2, n_loc * n2, MPI_DOUBLE, i, 0, global_mpi_data->comm, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(data, n3_loc, MPI_DOUBLE, 0, 0, global_mpi_data->comm);
    }
}


#else

int init_mpi() {
    if (global_mpi_data == NULL) {
        global_mpi_data = (mpi_data *)malloc(sizeof(mpi_data));
    }
    global_mpi_data->rank = 0;
    global_mpi_data->size = 1;

    return 0;
}

void cleanup_mpi() {
    if (global_mpi_data != NULL) {
        free(global_mpi_data);
        global_mpi_data = NULL;
    }
}

void exchange_bot_top(double *bot, double *top, double **bot_recv, double **top_recv) {
    *bot_recv = top;
    *top_recv = bot;
}

void bcast_double(double *buffer, long int size, int root) {
    // Do nothing
}

void allreduce_sum(double *buffer, long int size) {
    // Do nothing
}

void collect_grid_buffer(double *data, double *recv, int n) {
    if (data != recv) {
        memcpy(recv, data, n * n * n * sizeof(double));
    }
}

#endif

