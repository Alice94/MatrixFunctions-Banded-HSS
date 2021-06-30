/*=========================================================
 * fort.c
 * auxiliary routines for conversion between MATLAB and
 * FORTRAN complex data structures.
 *
 * Copyright 1984-2010 The MathWorks, Inc.
 *=======================================================*/
#include "mex.h"
#include "stddef.h"

/*
 * Convert MATLAB complex matrix to Fortran complex storage.
 * Z = mat2fort(X,ldz,ndz) converts MATLAB's mxArray X to Fortran's
 * complex*16 Z(ldz,ndz).  The parameters ldz and ndz determine the
 * storage allocated for Z, while mxGetM(X) and mxGetN(X) determine
 % the amount of data copied.
 */

double* mat2fort(
    const mxArray *X,
    ptrdiff_t ldz,
    ptrdiff_t ndz
    )
{
    ptrdiff_t incz;
    size_t i,j;
    size_t m,n;
    int cmplxflag;
    double *Z,*xr,*xi,*zp;

    Z = (double *) mxCalloc(2*ldz*ndz, sizeof(double));
    xr = mxGetPr(X);
    xi = mxGetPi(X);

    m = mxGetM(X);
    n = mxGetN(X);
    zp = Z;
    incz = 2*(ldz-m);
    cmplxflag = (xi != NULL);
    for (j = 0; j < n; j++) {
        if (cmplxflag) {
            for (i = 0; i < m; i++) {
                *zp++ = *xr++;
                *zp++ = *xi++;
            }
        } else {
            for (i = 0; i < m; i++) {
                *zp++ = *xr++;
                zp++;
            }
        }
        zp += incz;
    }
    return(Z);
}


/*
 * Convert Fortran complex storage to MATLAB real and imaginary parts.
 * X = fort2mat(Z,ldz,m,n) copies Z to X, producing a complex mxArray
 * with mxGetM(X) = m and mxGetN(X) = n.
 */

mxArray* fort2mat(
    double *Z,
    ptrdiff_t ldz,
    mwSize m,
    mwSize n
    )
{
    size_t i,j;
    ptrdiff_t incz;
    double *xr,*xi,*zp;
    mxArray *X;

    X = mxCreateDoubleMatrix(m,n,mxCOMPLEX);
    xr = mxGetPr(X);
    xi = mxGetPi(X);
    zp = Z;
    incz = 2*(ldz-m);
    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            *xr++ = *zp++;
            *xi++ = *zp++;
        }
        zp += incz;
    }
    return(X);
}

