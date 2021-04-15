#include <iostream>
#include <omp.h>
#include <math.h>

using namespace std;

const double PI = 3.14159265358979323846;
const int n = 10000000;

double func1(double x)
{
    return (sqrt(x * (3 - x))) / (x + 1);
}

double Integr(double a, double b, int n)
{
    double t = omp_get_wtime();
    double sq[2];
    const double eps = 1E-6;
    const int n0 = 100000000;
    #pragma omp parallel num_threads(4)
    {
        int n = n0, k;
        double delta = 1;
        for (k = 0; delta > eps; n *= 2, k ^= 1)//побитовое исключающее или
        {
            double h = (b - a) / n;
            double s = 0.0;
            sq[k] = 0;
            // Ждем пока все потоки закончат обнуление sq[k]
            #pragma omp barrier
            #pragma omp for nowait
            for (int i = 0; i < n; i++)
            {
                s += func1(a + h * (i + 0.5));
            }
            #pragma omp atomic
            sq[k] += s * h;
            #pragma omp barrier
            if (n > n0)
            {
                delta = fabs(sq[k] - sq[k ^ 1]) / 3.0;
            }
        }
        #pragma omp master
        printf("Result Pi: %.12f; Runge rule: EPS %e, n %d\n", sq[k] * sq[k], eps, n / 2);
    }
    t = omp_get_wtime() - t;
    printf("Elapsed time (sec.): %.6f\n", t);
    return 0;
}

double getrand(unsigned int *seed)
{
    return (double)rand_r(seed) / RAND_MAX;
}

double func2(double x, double y)
{
    return exp((x + y) * (x + y));
}

double Monte_Carlo_Method()
{
    int in = 0;
    double s = 0;
#pragma omp parallel num_threads(8)
    {
        double s_loc = 0;
        int in_loc = 0;
        unsigned int seed = omp_get_thread_num();
#pragma omp for nowait
        for (int i = 0; i < n; i++)
        {
            double x = getrand(&seed); // x in [0, 1]
            double y = getrand(&seed); // y in [0, 1-x]
            if (y <= (1 - x) && x <= 1)
            {
                in_loc++;
                s_loc += func2(x, y);
            }
        }
#pragma omp atomic
        s += s_loc;
#pragma omp atomic
        in += in_loc;
    }
    double v = PI * in / n;
    double res = v * s / in;
    return res;
}

int main(int argc, char **argv)
{
    int q = 0;
    cout << "Choose method\n" << "1)Integr\n" << "2)Monte_Carlo_Method" << endl;
    cin >> q;
    if (q == 1)
    {
        const double a = 1.0;
        const double b = 1.2;
        const int n = 10000000;
        printf("Numerical integration: [%f, %f], n = %d\n", a, b, n);
        Integr(a, b, n);
    }
    else if (q == 2)
    {
        double t = omp_get_wtime();
        double s = Monte_Carlo_Method();
        t = omp_get_wtime() - t;
        printf("Result : %.12f\n", s);
        printf("Elapsed time (sec.): %.12f\n", t);
    }
    return 0;
}
