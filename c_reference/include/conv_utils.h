#ifndef __CONVLAYER_UTILS__
#define __CONVLAYER_UTILS__

#include <math.h>
#include <float.h>

int prepareLowRankConvMat(float* out, float* W1, float* W2, unsigned rank, unsigned I, unsigned J);

#endif