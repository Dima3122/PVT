#include <iostream>
#include <omp.h>
#include <inttypes.h>

using namespace std;
#define THREASHOLD 1000
#define SIZE 10000000

void partition(int *v, int &i, int &j, int low, int high)
{
    i = low;
    j = high;
    int pivot = v[(low + high) / 2];
    do
    {
        while (v[i] < pivot)
            i++;
        while (v[j] > pivot)
            j--;
        if (i <= j)
        {
            swap(v[i], v[j]);
            i++;
            j--;
        }
    } while (i <= j);
}

void quicksort(int *v, int low, int high)
{
    int i, j;
    partition(v, i, j, low, high);
    if (low < j)
    {
        quicksort(v, low, j);
    }
    if (i < high)
    {
        quicksort(v, i, high);
    }
}

void quicksort_tasks(int *v, int low, int high)
{
    int i, j;
    partition(v, i, j, low, high);
    if (high - low < THREASHOLD || (j - low < THREASHOLD || high - i < THREASHOLD))
    {
        if (low < j)
        {
            quicksort_tasks(v, low, j);
        }
        if (i < high)
        {
            quicksort_tasks(v, i, high);
        }
    }
    else
    {
        #pragma omp task untied
        {
            quicksort_tasks(v, low, j);
        }
        quicksort_tasks(v, i, high);
    }
}

void run_task(int *v, int low, int high)
{
    #pragma omp parallel num_threads(6)
    {
        #pragma omp single
        quicksort_tasks(v, low, high);
    }
}

void print_arr(int *arr, int n)
{
    for (int i = 0; i < n; i++)
    {
        cout << "arr[" << i << "] = " << arr[i] << " ";
    }
}

void init_matr(int *arr, int n)
{
    for (int i = 0; i < n; i++)
    {
        arr[i] = rand() % 100;
    }
}

int main()
{
    int *arr = new int[SIZE];
    init_matr(arr, SIZE);
    double t = omp_get_wtime();
    //quicksort(arr, 0, SIZE -1);
    t = omp_get_wtime() - t;
    printf("Elapsed time quick_sort(sec.): %.12f\n", t);
    init_matr(arr, SIZE);
    t = omp_get_wtime();
    run_task(arr, 0, SIZE - 1);
    t = omp_get_wtime() - t;
    printf("Elapsed time quick_sort(paralel)(sec.): %.12f\n", t);
    return 0;
}