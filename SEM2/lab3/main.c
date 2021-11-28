#include <inttypes.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void get_chunk(int a, int b, int commsize, int rank, int *lb, int *ub)
{
    int n = b - a + 1;
    int q = n / commsize;
    if (n % commsize)
    {
        q++;
    }
    int r = commsize * q - n;
    int chunk = q;
    // первым commsize - r процессам достанется на одну строку больше
    if (rank >= commsize - r)
    {
        chunk = q - 1;
    }
    *lb = a;
    if (rank > 0)
    {
        if (rank <= commsize - r)
        {
            *lb += q * rank;
        }
        else
        {
            *lb = q * (commsize - r) + (q - 1) * (rank - (commsize - r));
        }
    }
    *ub = *lb + chunk - 1;
}

void sgemv(float *a, float *b, float *c, int m, int n)
{
    int commsize, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int lb, ub;
    get_chunk(0, m - 1, commsize, rank, &lb, &ub);
    int rows_count = ub - lb + 1;

    for (int i = 0; i < rows_count; i++)
    {
        c[lb + i] = 0.0;
        for (int j = 0; j < n; j++)
        {
            c[lb + i] += a[i * n + j] * b[j];
        }
    }
    int *displs = malloc(sizeof(int) * commsize);
    int *rcounts = malloc(sizeof(int) * commsize);
    for (int i = 0; i < commsize; i++)
    {
        get_chunk(0, m - 1, commsize, i, &lb, &ub);
        rows_count = ub - lb + 1;
        rcounts[i] = rows_count;
        displs[i] = (i > 0) ? displs[i - 1] + rcounts[i - 1] : 0;
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, c, rcounts, displs, MPI_FLOAT, MPI_COMM_WORLD);
}

int main(int argc, char *argv[])
{
    const int n = 45000;
    const int m = 45000;
    int commsize, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t = MPI_Wtime();
    int lb, ub;
    get_chunk(0, m - 1, commsize, rank, &lb, &ub);
    int rows_count = ub - lb + 1;
    float *a = malloc(sizeof(float) * rows_count * n);
    float *b = malloc(sizeof(float) * n);
    float *c = malloc(sizeof(float) * m);

    for (int i = 0; i < rows_count; i++)
    {
        for (int j = 0; j < n; j++)
        {
            a[i * n + j] = lb + i + 1;
        }
    }
    for (int j = 0; j < n; j++)
    {
        b[j] = j + 1;
    }

    sgemv(a, b, c, m, n);

    t = MPI_Wtime() - t;

    if (rank == 0)
    {
        // Validation
        for (int i = 0; i < m; i++)
        {
            float r = (i + 1) * (n / 2.0 + pow(n, 2) / 2.0);
            if (fabs(c[i] - r) > 1E-6)
            {
                fprintf(stderr, "Validation failed: elem %d = %f (real value %f)\n", i, c[i], r);
                break;
            }
        }
        printf("DGEMV: matrix-vector product (c[m] = a[m, n] * b[n]; m = "
               "%d, n = %d)\n",
               m, n);
        printf("Memory used: %" PRIu64 " MiB\n", (uint64_t)(((double)m * n + m + n) * sizeof(float)) >> 20);
        double gflop = 2.0 * m * n * 1E-9;
        printf("Elapsed time (%d procs): %.6f sec.\n", commsize, t);
        printf("Performance: %.2f GFLOPS\n", gflop / t);
    }

    free(a);
    free(b);
    free(c);
    MPI_Finalize();
    return 0;
}