#include "kernel_macro.h"

//
//  Compute Y += alpha*X
//
void sgeaxpy(int           m,
        int           n,
        float        alpha,
        const float  *X,
        int           incRowX,
        int           incColX,
        float        *Y,
        int           incRowY,
        int           incColY)
{
    int i, j;


    if (alpha!=1.0) {
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                Y[i*incColY+j*incRowY] += alpha*X[i*incColX+j*incRowX];
            }
        }
    } else {
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                Y[i*incColY+j*incRowY] += X[i*incColX+j*incRowX];
            }
        }
    }
}

//
//  Compute X *= alpha
//
void sgescal(int     m,
        int     n,
        float  alpha,
        float  *X,
        int     incRowX,
        int     incColX)
{
    int i, j;

    if (alpha!=0.0) {
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                X[i*incColX+j*incRowX] *= alpha;
            }
        }
    } else {
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                X[i*incColX+j*incRowX] = 0.0;
            }
        }
    }
}
