#include <math.h>
#include "tools.hpp"

double malahidanDistance(double* vectorX, double* vectorY, int size)
{
    double sum = 0, temp;
    for(int i = 0; i < size; ++i)
    {
        temp = vectorX[i] - vectorY[i];
        sum += temp*temp;
    }
    return sum;
}