#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <stdlib.h>

void matrix_vector_product(double *a, double *b, double *c, int m, int n)
{
    for (int i = 0; i < m; i++) 
    {
        c[i] = 0.0;
        for (int j = 0; j < n; j++)
        {
            c[i] += a[i * n + j] * b[j];
        }
    }
}

void matrix_vector_product_omp(double *a, double *b, double *c, int m, int n)
{
    #pragma omp parallel num_threads(6)
    {
        int nthreads = omp_get_num_threads();//общее количество потоков
        int threadid = omp_get_thread_num();//получаем айди потока который у нас есть
        int items_per_thread = m / nthreads;//количество строк на поток
        int lb = threadid * items_per_thread;//номер стартовой ячейки
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);//номер конечной ячейки. 
        for (int i = lb; i <= ub; i++) 
        {
            c[i] = 0.0;
            for (int j = 0; j < n; j++)
            {
                c[i] += a[i * n + j] * b[j];
            }
        }
    }
}

int main(int argc, char **argv)
{
    int n,m;
    n = m = 10000;//размер матрицы
    double *a, *b, *c;
    // Allocate memory for 2-d array a[m, n]
    a = malloc(sizeof(*a) * m * n);
    b = malloc(sizeof(*b) * n);
    c = malloc(sizeof(*c) * m);
    for (int i = 0; i < m; i++) 
    {
        for (int j = 0; j < n; j++)
        {
            a[i * n + j] = i + j;
        }
    }
    for (int j = 0; j < n; j++)
    {
        b[j] = j;
    }
    double t = omp_get_wtime();
    matrix_vector_product_omp(a, b, c, m, n);
    t = omp_get_wtime() - t;
    printf("Elapsed time %f sec.\n" ,t);
    free(a);
    free(b);
    free(c);
return 0;
}