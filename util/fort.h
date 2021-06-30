/*=========================================================
 * fort.h
 * header file for auxilliary routines for conversion
 * between MATLAB and FORTRAN complex data structures.
 *
 * Copyright 1984-2008 The MathWorks, Inc.
 *=======================================================*/

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
    );

/*
 * Convert Fortran complex storage to MATLAB real and imaginary parts.
 * X = fort2mat(Z,ldz,m,n) copies Z to X, producing a complex mxArray
 * with mxGetM(X) = m and mxGetN(X) = n.
 */

mxArray* fort2mat(
    double *Z,
    ptrdiff_t ldz,
    ptrdiff_t m,
    ptrdiff_t n
    );


