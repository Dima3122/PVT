#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>

const double PI = 3.14159265358979323846;
const int n = 10000000;

double getrand()
{
    return (double)rand() / RAND_MAX; 
}

double func2(double x, double y)
{
    return exp(x-y);
}

int main(int argc, char **argv)
{
    int rank, commsize;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int points_per_proc = n / commsize;
    int lb = rank * points_per_proc;
    int ub = (rank == commsize - 1) ? (n - 1) : (lb + points_per_proc - 1);

    double s = 0;
    int in = 0;
    double t = MPI_Wtime();
    srand(rank);
    for (int i = lb; i < ub; i++)
    {
        double x = getrand(); // x in [0, 1]
        double y = getrand(); // y in [0, 1-x]
        if (y <= (1 - x) && x <= 1)
        {
            in++;
            s += func2(x, y);
        }
    }
    int gin = 0;
    double gsum = 0;
    double tmax = 0;
    MPI_Reduce(&in, &gin, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&s, &gsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    t = MPI_Wtime() - t;
    MPI_Reduce(&t, &tmax, 1,MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) 
    {
        double v = PI * in / n;
        double res = v * s / in;
        printf("Result :%lf \n", res);
        printf("Elapsed time (sec.): %.6f\n", t);
    }
    MPI_Finalize();

    return 0;
}