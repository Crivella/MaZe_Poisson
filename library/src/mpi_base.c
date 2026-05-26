#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>

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

void mpi_printf(const char *format, ...) {
    if (global_mpi_data->rank == 0) {
        va_list args;
        va_start(args, format);
        vprintf(format, args);
        va_end(args);
    }
}

void mpi_fprintf(FILE *fp, const char *format, ...) {
    if (global_mpi_data->rank == 0) {
        va_list args;
        va_start(args, format);
        vfprintf(fp, format, args);
        va_end(args);
    }
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

    return global_mpi_data->size;
}

void cleanup_mpi() {
    if (global_mpi_data != NULL) {
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

void mpi_grid_exchange_bot_top(double *grid, int size1, int size2) {
    // Skip loop communication if the processor is holding no data
    if (size1 == 0) {
        return;
    }
    long int n2 = size2 * size2;

    double *bot = grid;
    double *top = grid + (size1 - 1) * n2;
    double *bot_recv = bot - n2;
    double *top_recv = top + n2;

    if (global_mpi_data->size == 1) {
        memcpy(top_recv, bot, n2 * sizeof(double));
        memcpy(bot_recv, top, n2 * sizeof(double));
    } else {
        MPI_Sendrecv(
            top, n2, MPI_DOUBLE, global_mpi_data->next_rank, 0,
            bot_recv, n2, MPI_DOUBLE, global_mpi_data->prev_rank, 0,
            global_mpi_data->comm, MPI_STATUS_IGNORE
        );
        MPI_Sendrecv(
            bot, n2, MPI_DOUBLE, global_mpi_data->prev_rank, 0,
            top_recv, n2, MPI_DOUBLE, global_mpi_data->next_rank, 0,
            global_mpi_data->comm, MPI_STATUS_IGNORE
        );
    }
}

void allreduce_sum(double *buffer, long int count) {
    if (global_mpi_data->size > 1) {
        MPI_Allreduce(MPI_IN_PLACE, buffer, count, MPI_DOUBLE, MPI_SUM, global_mpi_data->comm);
    }
}

void allreduce_max(double *buffer, long int count) {
    if (global_mpi_data->size > 1) {
        MPI_Allreduce(MPI_IN_PLACE, buffer, count, MPI_DOUBLE, MPI_MAX, global_mpi_data->comm);
    }
}

void bcast_double(double *buffer, long int size, int root) {
    if (global_mpi_data->size > 1) {
        MPI_Bcast(buffer, size, MPI_DOUBLE, root, global_mpi_data->comm);
    }
}

void barrier() {
    if (global_mpi_data->size > 1) {
        MPI_Barrier(global_mpi_data->comm);
    }
}

void mpi_grid_collect_buffer(double *data, double *recv, int n) {
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

void mpi_grid_exchange_bot_top(double *grid, int size1, int size2) {
    long int n2 = size2 * size2;

    double *bot = grid;
    double *top = grid + (size1 - 1) * n2;
    double *bot_recv = bot - n2;
    double *top_recv = top + n2;

    memcpy(top_recv, bot, n2 * sizeof(double));
    memcpy(bot_recv, top, n2 * sizeof(double));

}

void bcast_double(double *buffer, long int size, int root) {
    // Do nothing
}

void allreduce_sum(double *buffer, long int size) {
    // Do nothing
}

void allreduce_max(double *buffer, long int size) {
    // Do nothing (serial build)
}

void barrier() {
    // Do nothing
}

void mpi_grid_collect_buffer(double *data, double *recv, int n) {
    if (data != recv) {
        memcpy(recv, data, n * n * n * sizeof(double));
    }
}

#endif

double * mpi_grid_allocate(int size1, int size2) {
    long int n2 = size2 * size2;

    double *data = (double *)malloc((size1 + 2) * n2 * sizeof(double));
    if (data == NULL) {
        fprintf(stderr, "Error: Unable to allocate memory for grid data\n");
        exit(EXIT_FAILURE);
    }

    return data + n2;
}

void mpi_grid_free(double *data, int n) {
    long int n2 = n * n;
    free(data - n2);
}
