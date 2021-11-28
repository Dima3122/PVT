#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int rank, commsize;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Request reqs[commsize * 2];
    MPI_Status stats[commsize * commsize * 2];
    int count = 1024;
    char *sendbuf = malloc(sizeof(*sendbuf) * count * commsize);
    char *recvbuf = malloc(sizeof(*recvbuf) * count * commsize);

    double t = MPI_Wtime();
    for (int i = 0; i < commsize; i++) 
    {
        MPI_Isend(sendbuf + i * count, count, MPI_CHAR, i, 0, MPI_COMM_WORLD, &reqs[i]);
        MPI_Irecv(recvbuf + i * count, count, MPI_CHAR, i, 0, MPI_COMM_WORLD, &reqs[commsize + i]);
    }
    MPI_Waitall(commsize * 2, reqs, stats);

    t = MPI_Wtime() - t;
    double tmax;
    MPI_Reduce(&t, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) 
    {
        printf("Tmax = %lf\n", tmax);
    }
    free(sendbuf);
    free(recvbuf);
    MPI_Finalize();
    return 0;
}
