#include <stdio.h>
#include <mpi.h>
#include <math.h>

const double PI = 3.14159265358979323846;
const int n = 10000000;

double func1(double x)
{
    return (1-exp(0.7/x))/(2+x);
}

int main(int argc, char **argv)
{
    const double a = 1.0;
    const double b = 2;
    const int n = 10000000;
    int rank, commsize;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int points_per_proc = n / commsize;
    int lb = rank * points_per_proc;
    int ub = (rank == commsize - 1) ? (n - 1) : (lb + points_per_proc - 1);
    
    double t = MPI_Wtime();
    
    double sum = 0.0;
    double h = (b - a) / n;
    for (int i = lb; i <= ub; i++)
    {
        sum += func1(a + h * (i + 0.5));
    }

    double gsum = 0.0;
    double tmax = 0.0;
    MPI_Reduce(&sum, &gsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    t = MPI_Wtime() - t;
    MPI_Reduce(&t, &tmax, 1,MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) 
    {
        gsum *= h;
        printf("Result Pi: %.12f; error %.12f\n", gsum * gsum, fabs(gsum - sqrt(PI)));
        printf("Elapsed time (sec.): %.6f\n", tmax);
    }
    MPI_Finalize();
    return 0;
}