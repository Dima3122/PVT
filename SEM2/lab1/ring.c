#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) 
{
    int rank, commsize;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    for (int k = 0; k < 3; k++) 
    {
        int count = 1 * pow(1024, k);
        char *buf = malloc(sizeof(*buf) * count * commsize);

        int next = (rank + 1) % commsize;
        int prev = (rank - 1 + commsize) % commsize;

        int pos_send = rank;
        int pos_recv = prev;

        double t = MPI_Wtime();
        for (int i = 0; i < commsize - 1; i++) 
        {
            MPI_Sendrecv(buf + pos_send * count, count, MPI_CHAR, next, 0, buf + pos_recv * count, count, MPI_CHAR, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            pos_send = (pos_send - 1 + commsize) % commsize;
            pos_recv = (pos_recv - 1 + commsize) % commsize;
        }
        t = MPI_Wtime() - t;
        if (rank == 0) 
        {
            printf("%d: message size %d, time %.6f\n", rank, count, t);
        }
        free(buf);
    }
    MPI_Finalize();
    return 0;
}
