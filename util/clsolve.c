/*
**	clsolve.c
**
**	mex file to solve a Cauchy-like linear system.
**	Computes also an estimate of rcond.
**	The code is a C translation, widely based on BLAS, 
**	of the matlab function clsolve.m.
**	See clsolve.m
**
**	Compile with
**		mex clsolve.c fort.c -lmwblas -lmwlapack
**
**	Antonio Arico' & Giuseppe Rodriguez, University of Cagliari, Italy
**	Email: {arico,rodriguez}@unica.it
**
**	Last revised January 21, 2010
*/

#include "mex.h"
#include "fort.h"
#include "blas.h"
#include "lapack.h"

/* these includes are here to avoid incompatibility with Matlab includes */
#include <math.h>
#include <complex.h>

#define PINT	ptrdiff_t	/* starting from Matlab 7.8 (R2009a) */
/* #define PINT	int		/* up to Matlab 7.7 (R2008b) */

#define DP	double *
#define PP	PINT *

#if !defined(max)
#define	max(A, B)	((A) > (B) ? (A) : (B))
#endif

#if !defined(min)
#define	min(A, B)	((A) < (B) ? (A) : (B))
#endif


/* functions declarations */
void zschurcnp( PINT n, PINT dr, PINT kb, complex *g, complex *h, complex *t,
	complex *s, complex *b, double *rcond);
void zschurcpp( PINT n, PINT dr, PINT kb, complex *g, complex *h, complex *t,
	complex *s, complex *b, double *rcond, mwSize *p);
void zschurctp( PINT n, PINT dr, PINT kb, complex *g, complex *h, complex *t,
	complex *s, complex *b, double *rcond, mwSize *p, mwSize *q);
void zschurcpps( PINT n, PINT dr, PINT kb, complex *g, complex *h, complex *t,
	complex *s, complex *b, double *rcond, mwSize *p, mwSize *q);
void zschurcsb( PINT n, PINT dr, PINT kb, complex *g, complex *h, complex *t,
	complex *s, complex *b, double *rcond, mwSize *p, mwSize *q);
void zschurcgu( PINT n, PINT dr, PINT kb, PINT jstep, complex *g, complex *h,
	complex *t, complex *s, complex *b, double *rcond, 
	mwSize *p, mwSize *q);
void zschurcguex( PINT n, PINT dr, PINT kb, PINT jstep, complex *g, complex *h,
	complex *t, complex *s, complex *b, double *rcond, 
	mwSize *p, mwSize *q);
void dschurcnp( PINT n, PINT dr, PINT kb, double *g, double *h, double *t,
	double *s, double *b, double *rcond);
void dschurcpp( PINT n, PINT dr, PINT kb, double *g, double *h, double *t,
	double *s, double *b, double *rcond, mwSize *p);
void dschurctp( PINT n, PINT dr, PINT kb, double *g, double *h, double *t,
	double *s, double *b, double *rcond, mwSize *p, mwSize *q);
void dschurcpps( PINT n, PINT dr, PINT kb, double *g, double *h, double *t,
	double *s, double *b, double *rcond, mwSize *p, mwSize *q);
void dschurcsb( PINT n, PINT dr, PINT kb, double *g, double *h, double *t,
	double *s, double *b, double *rcond, mwSize *p, mwSize *q);
void dschurcgu( PINT n, PINT dr, PINT kb, PINT jstep, double *g, double *h,
	double *t, double *s, double *b, double *rcond, mwSize *p, mwSize *q);
void dschurcguex( PINT n, PINT dr, PINT kb, PINT jstep, double *g, double *h,
	double *t, double *s, double *b, double *rcond, mwSize *p, mwSize *q);


void mexFunction(
	int		nlhs,
	mxArray		*plhs[],
	int		nrhs,
	const mxArray	*prhs[]
	)
{
	/* mxSize nod_t, nod_s, nod_g, nod_h, nod_b; */
	const mwSize *dim_g, *dim_h, *dim_t, *dim_s, *dim_b;
	complex *gc, *hc, *tc, *sc, *bc;
	double  *gr, *hr, *tr, *sr, *br, *xr;
	double rcond, *ptr;
	mwSize n, dr, kb, i, j, addpar;
	int piv, complesso, ok;
	mwSize *p, *q;
	mwSize mt, nt, ms, ns;
	mxArray *rhs[1], *lhs[2];
	char str[256];
	PINT ione=1, pdt1;

	/* check inputs */
	if( nrhs < 5 ) 
		mexErrMsgIdAndTxt( "drsolve:clsolve:tooFewArguments",
			"Too few input arguments.");

	if( nrhs < 6 ) 
		piv = 1;	/* partial pivoting is the default */
	else {
		ptr = mxGetPr( prhs[5]);
		if( ptr == NULL )
			piv = 1;	/* partial pivoting is the default */
		else
			piv = lround(*ptr);
	}

	/* decode additional parameters */
	addpar = piv / 10;
	piv = piv % 10;

	/*
	nod_g = mxGetNumberOfDimensions(plhs[0]);
	nod_h = mxGetNumberOfDimensions(plhs[1]);
	nod_t = mxGetNumberOfDimensions(plhs[2]);
	nod_s = mxGetNumberOfDimensions(plhs[3]);
	nod_b = mxGetNumberOfDimensions(plhs[4]);
	*/

	dim_g = mxGetDimensions(prhs[0]);
	dim_h = mxGetDimensions(prhs[1]);
	dim_t = mxGetDimensions(prhs[2]);
	dim_s = mxGetDimensions(prhs[3]);
	dim_b = mxGetDimensions(prhs[4]);

	/* check size of inputs */
	mt = max(dim_t[0],dim_t[1]);
	nt = min(dim_t[0],dim_t[1]);
	ms = max(dim_s[0],dim_s[1]);
	ns = min(dim_s[0],dim_s[1]);

	n  = max(dim_g[0],dim_h[0]);
	dr = min(dim_g[0],dim_h[0]);
	n  = max(n,mt);
	dr = min(dr,mt);
	n  = max(n,ms);
	dr = min(dr,ms);
	n  = max(n,dim_b[0]);
	dr = min(dr,dim_b[0]);
	if( (n > dr) || (dim_g[1] != dim_h[1]) || (nt != 1) || (ns != 1) )
		mexErrMsgIdAndTxt( "drsolve:clsolve:inconsistentDimensions",
			"Input arrays have inconsistent dimensions.");

	/*n = dim_g[0];*/	/* size of linear system */
	dr = dim_g[1];		/* displacement rank */
	kb = dim_b[1];		/* number of columns in RHS */

	if( n < dr )
		mexErrMsgIdAndTxt( "drsolve:clsolve:badGenerators",
			"Generators have too many columns.");

	if( mxIsComplex(prhs[0]) || mxIsComplex(prhs[1])
	   || mxIsComplex(prhs[2]) || mxIsComplex(prhs[3])
	   || mxIsComplex(prhs[4]) )
		complesso = 1;	/* complex data */
	else
		complesso = 0;	/* real data */

	/* allocate and init permutation vectors */
	p = mxCalloc( n, sizeof(mwSize));
	q = mxCalloc( n, sizeof(mwSize));
	for( i=0; i<n; i++) {
		p[i] = i+1;
		q[i] = i+1;
	}
	
	/*
	g = (complex *) mat2fort( prhs[0], dim_g[0], dim_g[1]);
	h = (complex *) mat2fort( prhs[1], dim_h[0], dim_h[1]);
	t = (complex *) mat2fort( prhs[2], dim_t[0], dim_t[1]);
	s = (complex *) mat2fort( prhs[3], dim_s[0], dim_s[1]);
	b = (complex *) mat2fort( prhs[4], dim_b[0], dim_b[1]);
	*/

	if( complesso ) {
		/* allocate and populate arrays */
		gc   = (complex *) mat2fort( prhs[0], (PINT)n, (PINT)dr);
		hc   = (complex *) mat2fort( prhs[1], (PINT)n, (PINT)dr);
		tc   = (complex *) mat2fort( prhs[2], (PINT)dim_t[0],
			(PINT)dim_t[1]);
		if( piv == 2 ) {
			/* [s, q] = sort(s); */
			rhs[0] = (mxArray *) prhs[3];
			mexCallMATLAB( 2, lhs, 1, rhs, "sort");
			sc = (complex *) mat2fort( lhs[0], (PINT)dim_s[0],
				(PINT)dim_s[1]);
			xr = mxGetPr( lhs[1]);
			for( i=0; i<n; i++) 
				q[i] = round(xr[i]);
			mxDestroyArray( lhs[1]);
			mxDestroyArray( lhs[0]);
			/* H = H(:,q); */
			bc = mxCalloc( n, sizeof(complex));
			for( j=0; j<dr; j++) {
				for( i=0; i<n; i++)
					bc[i] = hc[q[i]-1+n*j];
				pdt1 = (PINT) n;
				zcopy( &pdt1, (DP)bc, &ione, (DP)(hc+n*j),
					&ione);
			}
			mxFree(bc);
		} else
			sc = (complex *) mat2fort( prhs[3], (PINT)dim_s[0],
				(PINT)dim_s[1]);
		bc   = (complex *) mat2fort( prhs[4], (PINT)n, (PINT)kb);

		/* conjugate second generator (not H=H') */
		for( i=0; i<n*dr; i++)
			*(hc+i) = conj(*(hc+i));

		/* check reconstructability */
		ok = 1;
		for( i=0; i<n && ok; i++)
			for( j=0; j<n && ok; j++)
				if( tc[i] == sc[j] )
					ok = 0;
		if( !ok )
			mexErrMsgIdAndTxt( "drsolve:clsolve:notReconstructable",
				"Partially reconstructable matrix.");

		/* check that there are no repetitions in s */
		ok = 1;
		if( piv != 2 )
			for( i=0; i<n && ok; i++)
				for( j=0; j<i && ok; j++)
					if( sc[i] == sc[j] )
						ok = 0;
		if( !ok )
			mexWarnMsgIdAndTxt( "drsolve:clsolve:multipleS",
				"Vector d2 has multiple entries, try piv=2.");

		switch( piv ) {
		case 0:
			zschurcnp( (PINT)n, (PINT)dr, (PINT)kb, gc, hc, tc, sc,
				bc, &rcond);
			break;
		case 1:
			zschurcpp( (PINT)n, (PINT)dr, (PINT)kb, gc, hc, tc, sc,
				bc, &rcond, p);
			break;
		case 2:
			zschurcpps( (PINT)n, (PINT)dr, (PINT)kb, gc, hc, tc,
				sc, bc, &rcond, p, q);
			break;
		case 3:
			zschurcsb( (PINT)n, (PINT)dr, (PINT)kb, gc, hc, tc, sc,
				bc, &rcond, p, q);
			break;
		case 4:
			zschurcgu( (PINT)n, (PINT)dr, (PINT)kb, (PINT)addpar,
				gc, hc, tc, sc, bc, &rcond, p, q);
			break;
		case 5:
			zschurctp( (PINT)n, (PINT)dr, (PINT)kb, gc, hc, tc, sc,
				bc, &rcond, p, q);
			break;
		case 6:
			zschurcguex( (PINT)n, (PINT)dr, (PINT)kb, (PINT)addpar,
				gc, hc, tc, sc, bc, &rcond, p, q);
			break;
		default:
			mexErrMsgIdAndTxt( 
				"drsolve:clsolve:pivotingNotSupported",
				"This kind of pivoting is unsupported.");
		}

		/* allocate and copy solution */
		plhs[0] = fort2mat( (double *)bc, (PINT)n, (PINT)n, (PINT)kb);

		mxFree(bc);
		mxFree(hc);
		mxFree(gc);
		mxFree(sc);
		mxFree(tc);
	} else {
		/* allocate and populate arrays */
		gr   = (double *) mxCalloc( n*dr, sizeof(double));
			xr = mxGetPr( prhs[0]);  pdt1 = n*dr;
			dcopy( &pdt1, xr, &ione, gr, &ione);
		hr   = (double *) mxCalloc( n*dr, sizeof(double));
			xr = mxGetPr( prhs[1]);  
			dcopy( &pdt1, xr, &ione, hr, &ione);
		tr   = (double *) mxCalloc( n, sizeof(double));
			xr = mxGetPr( prhs[2]);
			pdt1 = (PINT) n;
			dcopy( &pdt1, xr, &ione, tr, &ione);
		sr   = (double *) mxCalloc( n, sizeof(double));
		if( piv == 2 ) {
			/* [s, q] = sort(s); */
			rhs[0] = (mxArray *) prhs[3];
			mexCallMATLAB( 2, lhs, 1, rhs, "sort");
			xr = mxGetPr( lhs[0]);
			pdt1 = (PINT) n;
			dcopy( &pdt1, xr, &ione, sr, &ione);
			xr = mxGetPr( lhs[1]);
			for( i=0; i<n; i++) {
				q[i] = round(xr[i]);
			}
			mxDestroyArray( lhs[1]);
			mxDestroyArray( lhs[0]);
			/* H = H(:,q); */
			br = mxCalloc( n, sizeof(double));
			for( j=0; j<dr; j++) {
				for( i=0; i<n; i++)
					br[i] = hr[q[i]-1+n*j];
				pdt1 = (PINT) n;
				dcopy( &pdt1, br, &ione, hr+n*j, &ione);
			}
			mxFree(br);
		} else {
			xr = mxGetPr( prhs[3]);
			pdt1 = (PINT) n;
			dcopy( &pdt1, xr, &ione, sr, &ione);
		}
		br   = (double *) mxCalloc( n*kb, sizeof(double));
			xr = mxGetPr( prhs[4]);  pdt1 = n*kb;
			dcopy( &pdt1, xr, &ione, br, &ione);

		/* check reconstructability */
		ok = 1;
		for( i=0; i<n && ok; i++)
			for( j=0; j<n && ok; j++)
				if( tr[i] == sr[j] )
					ok = 0;
		if( !ok )
			mexErrMsgIdAndTxt( "drsolve:clsolve:notReconstructable",
				"Partially reconstructable matrix.");

		/* check that there are no repetitions in s */
		ok = 1;
		if( piv != 2 )
			for( i=0; i<n && ok; i++)
				for( j=0; j<i && ok; j++)
					if( sr[i] == sr[j] )
						ok = 0;
		if( !ok )
			mexWarnMsgIdAndTxt( "drsolve:clsolve:multipleS",
				"Vector d2 has multiple entries, try piv=2.");

		switch( piv ) {
		case 0:
			dschurcnp( (PINT)n, (PINT)dr, (PINT)kb, gr, hr, tr, sr,
				br, &rcond);
			break;
		case 1:
			dschurcpp( (PINT)n, (PINT)dr, (PINT)kb, gr, hr, tr, sr,
				br, &rcond, p);
			break;
		case 2:
			dschurcpps( (PINT)n, (PINT)dr, (PINT)kb, gr, hr, tr,
				sr, br, &rcond, p, q);
			break;
		case 3:
			dschurcsb( (PINT)n, (PINT)dr, (PINT)kb, gr, hr, tr, sr,
				br, &rcond, p, q);
			break;
		case 4:
			dschurcgu( (PINT)n, (PINT)dr, (PINT)kb, (PINT)addpar,
				gr, hr, tr, sr, br, &rcond, p, q);
			break;
		case 5:
			dschurctp( (PINT)n, (PINT)dr, (PINT)kb, gr, hr, tr, sr,
				br, &rcond, p, q);
			break;
		case 6:
			dschurcguex( (PINT)n, (PINT)dr, (PINT)kb, (PINT)addpar,
				gr, hr, tr, sr, br, &rcond, p, q);
			break;
		default:
			mexErrMsgIdAndTxt(
				"drsolve:clsolve:pivotingNotSupported",
				"This kind of pivoting is unsupported.");
		}

		/* allocate and copy solution */
		plhs[0] = mxCreateDoubleMatrix( n, kb, mxREAL);
		xr = mxGetPr( plhs[0]);  pdt1 = n*kb;
		dcopy( &pdt1, br, &ione, xr, &ione);

		mxFree(br);
		mxFree(hr);
		mxFree(gr);
		mxFree(sr);
		mxFree(tr);
	}

	/* if rcondu < eps
		warning('drsolve:clsolve:badConditioning', ...
			['Matrix is close to singular or badly scaled.\n', ...
			sprintf('         Results may be inaccurate. RCONDU = %g.',rcondu)])
	end */
	if( rcond < mxGetEps() ) {
		sprintf( str, "Matrix is close to singular or badly scaled.\n         Results may be inaccurate. RCONDU = %g.", rcond);
		mexWarnMsgIdAndTxt( "drsolve:clsolve:badConditioning", str);
	}

	if( nlhs > 1 )
		plhs[1] = mxCreateDoubleScalar(rcond);

	if( nlhs > 2 ) {
		plhs[2] = mxCreateDoubleMatrix( n, 1, mxREAL);
		xr = mxGetPr( plhs[2]);
		for( i=0; i<n; i++)
			xr[i] = p[i];
	}

	if( nlhs > 3 ) {
		plhs[3] = mxCreateDoubleMatrix( n, 1, mxREAL);
		xr = mxGetPr( plhs[3]);
		for( i=0; i<n; i++)
			xr[i] = q[i];
	}

	mxFree(q);
	mxFree(p);

}


/* no pivoting - complex */
void zschurcnp( PINT n, PINT dr, PINT kb, complex *g, complex *h, complex *t,
	complex *s, complex *b, double *rcond)
{
	PINT k, i, len;
	complex *l, *u1, *temp, pivot, scal;
	double uinvnm=0., *ucolsum;
	char notr[2] = {'N','\0'}, tran[2] = {'T','\0'}, cotr[2] = {'C','\0'};
	char lotr[2] = {'L','\0'}, nut[2] = {'N','\0'};
	complex one=1., minusone=-1., zero=0.;
	PINT ione=1, izero=0;

	l = mxCalloc( n, sizeof(complex));
	u1 = mxCalloc( n, sizeof(complex));
	ucolsum = mxCalloc( n, sizeof(double));
	temp = mxCalloc( max(n,kb), sizeof(complex));

	for( k=0; k<n; k++) {
		/* l = (G*H(:,k)) ./ ([s(1:k-1);t(k:n)]-s(k)); */
		zgemv( notr, &n, &dr, (DP)&one, (DP)g, &n, (DP)(h+k), &n,
			(DP)&zero, (DP)l, &ione);
		len = k;
		zcopy( &len, (DP)s, &ione, (DP)temp, &ione);
		len = n-k;
		zcopy( &len, (DP)(t+k), &ione, (DP)(temp+k), &ione);
		zaxpy( &n, (DP)&minusone, (DP)(s+k), &izero, (DP)temp, &ione);
		ztbsv( lotr, notr, nut, &n, &izero, (DP)temp, &ione, (DP)l,
			&ione);
		/* pivot = l(k); */
		pivot = l[k];
		/* l(k)  = -1; */
		l[k] = minusone;
		if( (creal(pivot) == 0.) && (cimag(pivot) == 0.) )
			mexWarnMsgIdAndTxt(
				"drsolve:clsolve:nullDiagonalElement",
			"Null diagonal element. Try to activate pivoting.");
		/* u1(1:n-k) = (G(k,:)*H(:,k+1:n)) ./ (t(k)-s(k+1:n).'); */
		len = n-k-1;
		zgemv( notr, &len, &dr, (DP)&one, (DP)(h+k+1), &n, (DP)(g+k),
			&n, (DP)&zero, (DP)u1, &ione);
		zcopy( &len, (DP)(t+k), &izero, (DP)temp, &ione);
		zaxpy( &len, (DP)&minusone, (DP)(s+k+1), &ione, (DP)temp,
			&ione);
		ztbsv( lotr, notr, nut, &len, &izero, (DP)temp, &ione, (DP)u1,
			&ione);
		/* g1_pivot = G(k,:)/pivot; */
		zcopy( &dr, (DP)(g+k), &n, (DP)temp, &ione);
		/* G(k,:) = 0; */
		zcopy( &dr, (DP)&zero, &izero, (DP)(g+k), &n);
		/* G = G-l*g1_pivot; */
		scal = minusone / pivot;
		zgeru( &n, &dr, (DP)&scal, (DP)l, &ione, (DP)temp, &ione,
			(DP)g, &n);
		/* g2_pivot = B(k,:)/pivot; */
		zcopy( &kb, (DP)(b+k), &n, (DP)temp, &ione);
		/* B(k,:) = 0; */
		zcopy( &kb, (DP)&zero, &izero, (DP)(b+k), &n);
		/* B = B-l*g2_pivot; */
		zgeru( &n, &kb, (DP)&scal, (DP)l, &ione, (DP)temp, &ione,
			(DP)b, &n);
		/* h1_pivot = H(:,k)/pivot; */
		zcopy( &dr, (DP)(h+k), &n, (DP)temp, &ione);
		/* H(:,k+1:n) = H(:,k+1:n) - h1_pivot*u1(1:n-k); */
		len = n-k-1;
		zgeru( &len, &dr, (DP)&scal, (DP)u1, &ione, (DP)temp, &ione,
			(DP)(h+k+1), &n);
		/* uinvnm = max([uinvnm; ...
		   sum(abs(real(l(1:k)))+abs(imag(l(1:k)))) / abs(pivot) ]); */
		len = k+1;
		uinvnm = max( uinvnm, dzasum( &len, (DP)l, &ione)/fabs(pivot));
		/* ucolsum(k:n) = ucolsum(k:n) + ...
			[ abs(real(pivot))+abs(imag(pivot)), ...
			abs(real(u1(1:n-k)))+abs(imag(u1(1:n-k))) ]; */
		ucolsum[k] += fabs(creal(pivot))+fabs(cimag(pivot));
		for( i=0; i<n-k-1; i++)
			ucolsum[k+i+1] += fabs(creal(u1[i]))+fabs(cimag(u1[i]));
	}

	/* unorm = max(ucolsum); */
	k = idamax( &n, ucolsum, &ione) - 1;
	/* rcond = 1/unorm/uinvnm; */
	*rcond = one/ucolsum[k]/uinvnm;

	mxFree(temp);
	mxFree(ucolsum);
	mxFree(u1);
	mxFree(l);
}


/* partial pivoting - complex */
void zschurcpp( PINT n, PINT dr, PINT kb, complex *g, complex *h, complex *t,
	complex *s, complex *b, double *rcond, mwSize *p)
{
	PINT imax, k, i, len;
	complex *l, *u1, *temp, pivot, scal;
	double uinvnm=0., *ucolsum;
	char notr[2] = {'N','\0'}, tran[2] = {'T','\0'}, cotr[2] = {'C','\0'};
	char lotr[2] = {'L','\0'}, nut[2] = {'N','\0'};
	complex one=1., minusone=-1., zero=0.;
	PINT ione=1, izero=0;

	l = mxCalloc( n, sizeof(complex));
	u1 = mxCalloc( n, sizeof(complex));
	ucolsum = mxCalloc( n, sizeof(double));
	temp = mxCalloc( max(n,kb), sizeof(complex));

	for( k=0; k<n; k++) {
		/* l = (G*H(:,k)) ./ ([s(1:k-1);t(k:n)]-s(k)); */
		zgemv( notr, &n, &dr, (DP)&one, (DP)g, &n, (DP)(h+k), &n,
			(DP)&zero, (DP)l, &ione);
		len = k;
		zcopy( &len, (DP)s, &ione, (DP)temp, &ione);
		len = n-k;
		zcopy( &len, (DP)(t+k), &ione, (DP)(temp+k), &ione);
		zaxpy( &n, (DP)&minusone, (DP)(s+k), &izero, (DP)temp, &ione);
		ztbsv( lotr, notr, nut, &n, &izero, (DP)temp, &ione, (DP)l,
			&ione);
		/* [lmax imax] = max( abs(real(l(k:n)))+abs(imag(l(k:n))) ); */
		len = n-k;
		imax = izamax( &len, (DP)(l+k), &ione) - 1;
		/* if lmax == 0
			warning('drsolve:clsolve', ...
			'Matrix is singular to working precision.')
		end */
		if( (creal(l[k+imax]) == 0.) && (cimag(l[k+imax]) == 0.) )
			mexWarnMsgIdAndTxt( "drsolve:clsolve:singularMatrix",
				"Matrix is singular to working precision.");
		/* if imax > 1
			imax  = imax+(k-1);
			l([k imax])   = l([imax k]);
			t([k imax])   = t([imax k]);
			G([k imax],:) = G([imax k],:);
			B([k imax],:) = B([imax k],:);
			p([k imax])   = p([imax k]);
		    end
		    pivot = l(k); */
		if( imax > 0 ) {
			imax += k;
			pivot = l[imax];  l[imax] = l[k];
			scal = t[k];  t[k] = t[imax];  t[imax] = scal;
			zswap( &dr, (DP)(g+k), &n, (DP)(g+imax), &n);
			zswap( &kb, (DP)(b+k), &n, (DP)(b+imax), &n);
			i = p[k];  p[k] = p[imax];  p[imax] = i;
		} else
			pivot = l[k];
		/* l(k) = -1; */
		l[k] = minusone;
		/* u1(1:n-k) = (G(k,:)*H(:,k+1:n)) ./ (t(k)-s(k+1:n).'); */
		len = n-k-1;
		zgemv( notr, &len, &dr, (DP)&one, (DP)(h+k+1), &n, (DP)(g+k),
			&n, (DP)&zero, (DP)u1, &ione);
		zcopy( &len, (DP)(t+k), &izero, (DP)temp, &ione);
		zaxpy( &len, (DP)&minusone, (DP)(s+k+1), &ione, (DP)temp,
			&ione);
		ztbsv( lotr, notr, nut, &len, &izero, (DP)temp, &ione, (DP)u1,
			&ione);
		/* g1_pivot = G(k,:)/pivot; */
		zcopy( &dr, (DP)(g+k), &n, (DP)temp, &ione);
		/* G(k,:) = 0; */
		zcopy( &dr, (DP)&zero, &izero, (DP)(g+k), &n);
		/* G = G-l*g1_pivot; */
		scal = minusone / pivot;
		zgeru( &n, &dr, (DP)&scal, (DP)l, &ione, (DP)temp, &ione,
			(DP)g, &n);
		/* g2_pivot = B(k,:)/pivot; */
		zcopy( &kb, (DP)(b+k), &n, (DP)temp, &ione);
		/* B(k,:) = 0; */
		zcopy( &kb, (DP)&zero, &izero, (DP)(b+k), &n);
		/* B = B-l*g2_pivot; */
		zgeru( &n, &kb, (DP)&scal, (DP)l, &ione, (DP)temp, &ione,
			(DP)b, &n);
		/* h1_pivot = H(:,k)/pivot; */
		zcopy( &dr, (DP)(h+k), &n, (DP)temp, &ione);
		/* H(:,k+1:n) = H(:,k+1:n) - h1_pivot*u1(1:n-k); */
		len = n-k-1;
		zgeru( &len, &dr, (DP)&scal, (DP)u1, &ione, (DP)temp, &ione,
			(DP)(h+k+1), &n);
		/* uinvnm = max([uinvnm; ...
		   sum(abs(real(l(1:k)))+abs(imag(l(1:k)))) / abs(pivot) ]); */
		len = k+1;
		uinvnm = max( uinvnm, dzasum( &len, (DP)l, &ione)/fabs(pivot));
		/* ucolsum(k:n) = ucolsum(k:n) + ...
			[ abs(real(pivot))+abs(imag(pivot)), ...
			abs(real(u1(1:n-k)))+abs(imag(u1(1:n-k))) ]; */
		ucolsum[k] += fabs(creal(pivot))+fabs(cimag(pivot));
		for( i=0; i<n-k-1; i++)
			ucolsum[k+i+1] += fabs(creal(u1[i]))+fabs(cimag(u1[i]));
	}

	/* unorm = max(ucolsum); */
	k = idamax( &n, ucolsum, &ione) - 1;
	/* rcond = 1/unorm/uinvnm; */
	*rcond = one/ucolsum[k]/uinvnm;

	mxFree(temp);
	mxFree(ucolsum);
	mxFree(u1);
	mxFree(l);
}


/* partial pivoting with multiple s's - complex */
void zschurcpps( PINT n, PINT dr, PINT kb, complex *g, complex *h, complex *t,
	complex *s, complex *b, double *rcond, mwSize *p, mwSize *q)
{
	PINT imax=0, k, i, j, len, *i1, *i2;
	complex *l, *u1, *temp, pivot, scal;
	double uinvnm=0., *ucolsum;
	char notr[2] = {'N','\0'}, tran[2] = {'T','\0'}, cotr[2] = {'C','\0'};
	char lotr[2] = {'L','\0'}, nut[2] = {'N','\0'};
	complex one=1., minusone=-1., zero=0.;
	PINT ione=1, izero=0;

	l = mxCalloc( n, sizeof(complex));
	u1 = mxCalloc( n, sizeof(complex));
	ucolsum = mxCalloc( n, sizeof(double));
	temp = mxCalloc( max(n,kb), sizeof(complex));
	i1 = mxCalloc( n, sizeof(PINT));
	i2 = mxCalloc( n, sizeof(PINT));

	/* [ss, a, b] = unique(s,'first');
	[ss, c, d] = unique(s,'last');
	i1 = a(b);
	i2 = c(d);
	delta = max(diff([a; numel(s)+1])); */
	i1[0] = 0;  scal = s[0];
	for( i=1; i<n; i++)
		if( (creal(scal)==creal(s[i])) && (cimag(scal)==cimag(s[i])) )
			i1[i] = i1[i-1];
		else {
			len = i-1;
			for( k=i1[len]; k<i; k++)
				i2[k] = len;
			i1[i] = i;  scal = s[i];
			imax = max( imax, i-i1[len]);
		}
	len = n-1;
	for( k=i1[len]; k<n; k++)
		i2[k] = len;
	imax = max( imax, n-i1[len]);
	/* if delta > dr
                 error('drsolve:clsolve:singularMatrix', ...
                        'Matrix is singular to working precision.')
	end */
	if( imax > dr )
		mexErrMsgIdAndTxt( "drsolve:clsolve:singularMatrix",
			"Cauchy-like matrix is singular.");

	for( k=0; k<n; k++) {
		/* l(k:n) = (G(k:n,:) *H(:,k)) ./ (t(k:n) -s(k)); */
		len = n-k;
		zgemv( notr, &len, &dr, (DP)&one, (DP)(g+k), &n, (DP)(h+k), &n,
			(DP)&zero, (DP)(l+k), &ione);
		zcopy( &len, (DP)(t+k), &ione, (DP)(temp+k), &ione);
		zaxpy( &len, (DP)&minusone, (DP)(s+k), &izero, (DP)(temp+k),
			&ione);
		ztbsv( lotr, notr, nut, &len, &izero, (DP)(temp+k), &ione, 
			(DP)(l+k), &ione);
		/* l(1:i1(k)-1) = (G(1:i1(k)-1,:)*H(:,k)) ./ 
		  				(s(1:i1(k)-1)-s(k)); */
		len = i1[k];
		if( len ) {
			zgemv( notr, &len, &dr, (DP)&one, (DP)g, &n, (DP)(h+k),
				&n, (DP)&zero, (DP)l, &ione);
			zcopy( &len, (DP)s, &ione, (DP)temp, &ione);
			zaxpy( &len, (DP)&minusone, (DP)(s+k), &izero,
				(DP)temp, &ione);
			ztbsv( lotr, notr, nut, &len, &izero, (DP)temp, &ione, 
				(DP)l, &ione);
		}
		/* if i1(k) < k
			l(i1(k):k-1) = H(k-i1(k),i1(k):k-1).';
		   end */
		if( i1[k] < k ) {
			len = k-i1[k];
			zcopy( &len, (DP)(h+(len-1)*n+i1[k]), &ione,
				(DP)(l+i1[k]), &ione);
		}
		/* [lmax imax] = max( abs(real(l(k:n)))+abs(imag(l(k:n))) ); */
		len = n-k;
		imax = izamax( &len, (DP)(l+k), &ione) - 1;
		/* if lmax == 0
			warning('drsolve:clsolve', ...
			'Matrix is singular to working precision.')
		   end */
		if( (creal(l[k+imax]) == 0.) && (cimag(l[k+imax]) == 0.) )
			mexWarnMsgIdAndTxt( "drsolve:clsolve:singularMatrix",
				"Matrix is singular to working precision.");
		/* if imax > 1
			imax  = imax+(k-1);
			l([k imax])   = l([imax k]);
			t([k imax])   = t([imax k]);
			G([k imax],:) = G([imax k],:);
			B([k imax],:) = B([imax k],:);
			p([k imax])   = p([imax k]);
		   end
		   pivot = l(k); */
		if( imax > 0 ) {
			imax += k;
			pivot = l[imax];  l[imax] = l[k];
			scal = t[k];  t[k] = t[imax];  t[imax] = scal;
			zswap( &dr, (DP)(g+k), &n, (DP)(g+imax), &n);
			zswap( &kb, (DP)(b+k), &n, (DP)(b+imax), &n);
			i = p[k];  p[k] = p[imax];  p[imax] = i;
		} else
			pivot = l[k];
		/* l(k) = -1; */
		l[k] = minusone;
		/* u1(1:n-k) = (G(k,:)*H(:,k+1:n)) ./ (t(k)-s(k+1:n).'); */
		len = n-k-1;
		zgemv( notr, &len, &dr, (DP)&one, (DP)(h+k+1), &n, (DP)(g+k),
			&n, (DP)&zero, (DP)u1, &ione);
		zcopy( &len, (DP)(t+k), &izero, (DP)temp, &ione);
		zaxpy( &len, (DP)&minusone, (DP)(s+k+1), &ione, (DP)temp,
			&ione);
		ztbsv( lotr, notr, nut, &len, &izero, (DP)temp, &ione, (DP)u1,
			&ione);
		/* g1_pivot = G(k,:)/pivot; */
		zcopy( &dr, (DP)(g+k), &n, (DP)temp, &ione);
		/* G(k,:) = 0; */
		zcopy( &dr, (DP)&zero, &izero, (DP)(g+k), &n);
		/* G = G-l*g1_pivot; */
		scal = minusone / pivot;
		zgeru( &n, &dr, (DP)&scal, (DP)l, &ione, (DP)temp, &ione,
			(DP)g, &n);
		/* g2_pivot = B(k,:)/pivot; */
		zcopy( &kb, (DP)(b+k), &n, (DP)temp, &ione);
		/* B(k,:) = 0; */
		zcopy( &kb, (DP)&zero, &izero, (DP)(b+k), &n);
		/* B = B-l*g2_pivot; */
		zgeru( &n, &kb, (DP)&scal, (DP)l, &ione, (DP)temp, &ione,
			(DP)b, &n);
		/* h1_pivot = H(:,k)/pivot; */
		zcopy( &dr, (DP)(h+k), &n, (DP)temp, &ione);
		/* H(:,k+1:n) = H(:,k+1:n) - h1_pivot*u1(1:n-k); */
		len = n-k-1;
		zgeru( &len, &dr, (DP)&scal, (DP)u1, &ione, (DP)temp, &ione,
			(DP)(h+k+1), &n);
		/* if k < i2(k)
			pos = k-i1(k)+1;
			len = i2(k)-i1(k);
			H(pos:len,k)       = 0;
			H(pos:len,i1(k):k) = H(pos:len,i1(k):k) - 
					     u1(1:i2(k)-k).'*(l(i1(k):k).'/
					     pivot);
		   end */
		if( k < i2[k] ) {
			j = k-i1[k]+1;
			len = i2[k]-k;
			zcopy( &len, (DP)&zero, &izero, (DP)(h+(j-1)*n+k), &n);
			scal = minusone / pivot;
			zgeru( &j, &len, (DP)&scal, (DP)(l+i1[k]), &ione,
				(DP)u1, &ione, (DP)(h+(j-1)*n+i1[k]), &n);
		}
		/* uinvnm = max([uinvnm; ...
		   sum(abs(real(l(1:k)))+abs(imag(l(1:k)))) / abs(pivot) ]); */
		len = k+1;
		uinvnm = max( uinvnm, dzasum( &len, (DP)l, &ione)/fabs(pivot));
		/* ucolsum(k:n) = ucolsum(k:n) + ...
			[ abs(real(pivot))+abs(imag(pivot)), ...
			abs(real(u1(1:n-k)))+abs(imag(u1(1:n-k))) ]; */
		ucolsum[k] += fabs(creal(pivot))+fabs(cimag(pivot));
		for( i=0; i<n-k-1; i++)
			ucolsum[k+i+1] += fabs(creal(u1[i]))+fabs(cimag(u1[i]));
	}

	/* unorm = max(ucolsum); */
	k = idamax( &n, ucolsum, &ione) - 1;
	/* rcond = 1/unorm/uinvnm; */
	*rcond = one/ucolsum[k]/uinvnm;

	/* B(q,:) = B; */
	for( j=0; j<kb; j++) {
		for( i=0; i<n; i++)
			temp[q[i]-1] = b[i+n*j];
		zcopy( &n, (DP)temp, &ione, (DP)(b+n*j), &ione);
	}

	mxFree(i2);
	mxFree(i1);
	mxFree(temp);
	mxFree(ucolsum);
	mxFree(u1);
	mxFree(l);
}


/* Sweet&Brent's pivoting - complex */
void zschurcsb( PINT n, PINT dr, PINT kb, complex *g, complex *h, complex *t,
	complex *s, complex *b, double *rcond, mwSize *p, mwSize *q)
{
	PINT imax, jmax, k, i, j, len;
	complex *l, *u1, *temp, pivot, scal;
	double uinvnm=0., *ucolsum, p1, p2;
	char notr[2] = {'N','\0'}, tran[2] = {'T','\0'}, cotr[2] = {'C','\0'};
	char lotr[2] = {'L','\0'}, nut[2] = {'N','\0'};
	complex one=1., minusone=-1., zero=0.;
	PINT ione=1, izero=0;

	l = mxCalloc( n, sizeof(complex));
	u1 = mxCalloc( n, sizeof(complex));
	ucolsum = mxCalloc( n, sizeof(double));
	temp = mxCalloc( max(n,kb), sizeof(complex));

	for( k=0; k<n; k++) {
		/* l(k:n) = (G(k:n,:)*H(:,k)) ./ (t(k:n)-s(k)); */
		len = n-k;
		zgemv( notr, &len, &dr, (DP)&one, (DP)(g+k), &n, (DP)(h+k), &n,
			(DP)&zero, (DP)(l+k), &ione);
		zcopy( &len, (DP)(t+k), &ione, (DP)temp, &ione);
		zaxpy( &len, (DP)&minusone, (DP)(s+k), &izero, (DP)temp, &ione);
		ztbsv( lotr, notr, nut, &len, &izero, (DP)temp, &ione,
			(DP)(l+k), &ione);
		/* [p1,i1] = max(abs(real(l(k:n)))+abs(imag(l(k:n))));
		r_ind = i1+k-1; */
		imax = izamax( &len, (DP)(l+k), &ione) - 1;
		imax += k;
		p1 = fabs(creal(l[imax])) + fabs(cimag(l[imax]));
		/* u1(1:n-k) = (G(k,:)*H(:,k+1:n)) ./ (t(k)-s(k+1:n).');
		[p2,i2] = max(abs(real(u1(1:n-k)))+abs(imag(u1(1:n-k))));
		c_ind = i2+k; */
		len = n-k-1;
		if( len ) {
			zgemv( notr, &len, &dr, (DP)&one, (DP)(h+k+1), &n,
				(DP)(g+k), &n, (DP)&zero, (DP)u1, &ione);
			zcopy( &len, (DP)(t+k), &izero, (DP)temp, &ione);
			zaxpy( &len, (DP)&minusone, (DP)(s+k+1), &ione,
				(DP)temp, &ione);
			ztbsv( lotr, notr, nut, &len, &izero, (DP)temp, &ione, 
				(DP)u1, &ione);
			jmax = izamax( &len, (DP)u1, &ione) - 1;
			p2 = fabs(creal(u1[jmax])) + fabs(cimag(u1[jmax]));
			jmax = jmax+k+1;
		} else
			p2 = 0.;
		/* if ( max(p1,p2)==0 )
			error('drsolve:clsolve:singularMatrix', ...
			'Matrix is singular to working precision.') */
		if( max(p1,p2) == 0. )
			mexWarnMsgIdAndTxt( "drsolve:clsolve:singularMatrix",
				"Matrix is singular to working precision.");
		/* elseif ( p2>p1 )
			s(  [c_ind,k]) = s(  [k,c_ind]);
			H(:,[c_ind,k]) = H(:,[k,c_ind]);
			q(  [c_ind,k]) = q(  [k,c_ind]);
			%
			u1(i2) = l(k);                                 % update
			l(k:n) = ( G(k:n,:)*H(:,k) ) ./ (t(k:n)-s(k)); % recompute
		end */
		else if( p2>p1 ) {
			scal = s[k];  s[k] = s[jmax];  s[jmax] = scal;
			zswap( &dr, (DP)(h+k), &n, (DP)(h+jmax), &n);
			i = q[k];  q[k] = q[jmax];  q[jmax] = i;
			u1[jmax-k-1] = l[k];
			len = n-k;
			zgemv( notr, &len, &dr, (DP)&one, (DP)(g+k), &n,
				(DP)(h+k), &n, (DP)&zero, (DP)(l+k), &ione);
			zcopy( &len, (DP)(t+k), &ione, (DP)temp, &ione);
			zaxpy( &len, (DP)&minusone, (DP)(s+k), &izero,
				(DP)temp, &ione);
			ztbsv( lotr, notr, nut, &len, &izero, (DP)temp, &ione, 
				(DP)(l+k), &ione);
		} else {
		/* if r_ind ~= k
			l([r_ind,k]  ) = l([k,r_ind]  );
			t([r_ind,k]  ) = t([k,r_ind]  );
			G([r_ind,k],:) = G([k,r_ind],:);
			B([r_ind,k],:) = B([k,r_ind],:);
			p([r_ind,k]  ) = p([k,r_ind]  );
			u1(1:n-k) = (G(k,:)*H(:,k+1:n)) ./ (t(k)-s(k+1:n).');
		   end */
			if( imax > k ) {
				pivot = l[imax];  l[imax] = l[k];  l[k] = pivot;
				scal = t[k];  t[k] = t[imax];  t[imax] = scal;
				zswap( &dr, (DP)(g+k), &n, (DP)(g+imax), &n);
				zswap( &kb, (DP)(b+k), &n, (DP)(b+imax), &n);
				i = p[k];  p[k] = p[imax];  p[imax] = i;
				len = n-k-1;
				zgemv( notr, &len, &dr, (DP)&one, (DP)(h+k+1),
					&n, (DP)(g+k), &n, (DP)&zero, (DP)u1,
					&ione);
				zcopy( &len, (DP)(t+k), &izero, (DP)temp,
					&ione);
				zaxpy( &len, (DP)&minusone, (DP)(s+k+1), &ione,
					(DP)temp, &ione);
				ztbsv( lotr, notr, nut, &len, &izero, (DP)temp,
					&ione, (DP)u1, &ione);
			}
		}
		/* l(1:k-1) = (G(1:k-1,:)*H(:,k)) ./ (s(1:k-1)-s(k)); */
		if( k ) {
			zgemv( notr, &k, &dr, (DP)&one, (DP)g, &n, (DP)(h+k),
				&n, (DP)&zero, (DP)l, &ione);
			zcopy( &k, (DP)s, &ione, (DP)temp, &ione);
			zaxpy( &k, (DP)&minusone, (DP)(s+k), &izero, (DP)temp,
				&ione);
			ztbsv( lotr, notr, nut, &k, &izero, (DP)temp, &ione, 
				(DP)l, &ione);
		}
		/* pivot = l(k);
		   l(k) = -1; */
		pivot = l[k];
		l[k] = minusone;
		/* g1_pivot = G(k,:)/pivot; */
		zcopy( &dr, (DP)(g+k), &n, (DP)temp, &ione);
		/* G(k,:) = 0; */
		zcopy( &dr, (DP)&zero, &izero, (DP)(g+k), &n);
		/* G = G-l*g1_pivot; */
		scal = minusone / pivot;
		zgeru( &n, &dr, (DP)&scal, (DP)l, &ione, (DP)temp, &ione,
			(DP)g, &n);
		/* g2_pivot = B(k,:)/pivot; */
		zcopy( &kb, (DP)(b+k), &n, (DP)temp, &ione);
		/* B(k,:) = 0; */
		zcopy( &kb, (DP)&zero, &izero, (DP)(b+k), &n);
		/* B = B-l*g2_pivot; */
		zgeru( &n, &kb, (DP)&scal, (DP)l, &ione, (DP)temp, &ione,
			(DP)b, &n);
		/* h1_pivot = H(:,k)/pivot; */
		zcopy( &dr, (DP)(h+k), &n, (DP)temp, &ione);
		/* H(:,k+1:n) = H(:,k+1:n) - h1_pivot*u1(1:n-k); */
		len = n-k-1;
		if( len )
			zgeru( &len, &dr, (DP)&scal, (DP)u1, &ione, (DP)temp,
				&ione, (DP)(h+k+1), &n);
		/* uinvnm = max([uinvnm; ...
		   sum(abs(real(l(1:k)))+abs(imag(l(1:k)))) / abs(pivot) ]); */
		len = k+1;
		uinvnm = max( uinvnm, dzasum( &len, (DP)l, &ione)/fabs(pivot));
		/* ucolsum(k:n) = ucolsum(k:n) + ...
			[ abs(real(pivot))+abs(imag(pivot)), ...
			abs(real(u1(1:n-k)))+abs(imag(u1(1:n-k))) ]; */
		ucolsum[k] += fabs(creal(pivot))+fabs(cimag(pivot));
		for( i=0; i<n-k-1; i++)
			ucolsum[k+i+1] += fabs(creal(u1[i]))+fabs(cimag(u1[i]));
	}

	/* unorm = max(ucolsum); */
	k = idamax( &n, ucolsum, &ione) - 1;
	/* rcond = 1/unorm/uinvnm; */
	*rcond = one/ucolsum[k]/uinvnm;

	/* B(q,:) = B; */
	for( j=0; j<kb; j++) {
		for( i=0; i<n; i++)
			temp[q[i]-1] = b[i+n*j];
		zcopy( &n, (DP)temp, &ione, (DP)(b+n*j), &ione);
	}

	mxFree(temp);
	mxFree(ucolsum);
	mxFree(u1);
	mxFree(l);
}


/* Gu's pivoting - complex */
void zschurcgu( PINT n, PINT dr, PINT kb, PINT jstep, complex *g, complex *h,
	complex *t, complex *s, complex *b, double *rcond, mwSize *p, mwSize *q)
{
	PINT imax, k, i, j, len, ltemp, info;
	complex *l, *u1, *tau, *temp, pivot, scal;
	double uinvnm=0., *ucolsum, nmax, tt;
	char notr[2] = {'N','\0'}, tran[2] = {'T','\0'}, cotr[2] = {'C','\0'};
	char lotr[2] = {'L','\0'}, nut[2] = {'N','\0'};
	char uptr[2] = {'U','\0'}, right[2] = {'R','\0'};
	complex one=1., minusone=-1., zero=0.;
	PINT ione=1, izero=0;

	if( !jstep )
		jstep = 10;	/* default value */

	l = mxCalloc( n, sizeof(complex));
	u1 = mxCalloc( n, sizeof(complex));
	ucolsum = mxCalloc( n, sizeof(double));
	tau = mxCalloc( dr, sizeof(complex));

	/* get optimal dimension of work array */
	ltemp = -1;
	zgeqrf( &n, &dr, (DP)g, &n, (DP)tau, (DP)&scal, &ltemp, &info);
	if( info )  mexErrMsgTxt( "Error in zschurcgu/zgeqrf(init).");
	ltemp = (PINT) scal;
	ltemp = max(ltemp,n);
	ltemp = max(ltemp,kb);
	temp = mxCalloc( ltemp, sizeof(complex));

	for( k=0; k<n; k++) {
		/* if (mod(k,Jmax_step) == 1) && (k <= n-dr+1)
			[ G(k:n,:) R ] = qr(G(k:n,:),0);
			G(1:k-1,:) = G(1:k-1,:) / R;
			H(:,k:n) = R*H(:,k:n);
			[ nmax c_ind ] = max(sum( abs(real(H(:,k:end)))...
						+ abs(imag(H(:,k:end))), 1));
			c_ind = c_ind+(k-1);
			if c_ind ~= k
				s(  [c_ind,k]) = s(  [k,c_ind]);
				H(:,[c_ind,k]) = H(:,[k,c_ind]);
				q(  [c_ind,k]) = q(  [k,c_ind]);
			end
		end */
		if( ((k % jstep) == 0) && (k <= n-dr) ) {
			len = n-k;
			zgeqrf( &len, &dr, (DP)(g+k), &n, (DP)tau, (DP)temp,
				&ltemp, &info);
			if( info )  mexErrMsgTxt( "Error in zschurcgu/zgeqrf.");
			ztrmm( right, uptr, tran, nut, &len, &dr, (DP)&one,
				(DP)(g+k), &n, (DP)(h+k), &n);
			ztrsm( right, uptr, notr, nut, &k, &dr, (DP)&one, 
				(DP)(g+k), &n, (DP)g, &n);
			zungqr( &len, &dr, &dr, (DP)(g+k), &n, (DP)tau, 
				(DP)temp, &ltemp, &info);
			if( info )  mexErrMsgTxt( "Error in zschurcgu/zungqr.");
			nmax = dzasum( &dr, (DP)(h+k), &n);
			j = k;
			for( i=k+1; i<n; i++) {
				tt = dzasum( &dr, (DP)(h+i), &n);
				if( tt > nmax ) {
					nmax = tt;
					j = i;
				}
			}
			if( j > k ) {
				scal = s[k];  s[k] = s[j];  s[j] = scal;
				zswap( &dr, (DP)(h+k), &n, (DP)(h+j), &n);
				i = q[k];  q[k] = q[j];  q[j] = i;
			}
		}
		/* l = (G*H(:,k)) ./ ([s(1:k-1);t(k:n)]-s(k)); */
		zgemv( notr, &n, &dr, (DP)&one, (DP)g, &n, (DP)(h+k), &n,
			(DP)&zero, (DP)l, &ione);
		len = k;
		zcopy( &len, (DP)s, &ione, (DP)temp, &ione);
		len = n-k;
		zcopy( &len, (DP)(t+k), &ione, (DP)(temp+k), &ione);
		zaxpy( &n, (DP)&minusone, (DP)(s+k), &izero, (DP)temp, &ione);
		ztbsv( lotr, notr, nut, &n, &izero, (DP)temp, &ione, (DP)l,
			&ione);
		/* [lmax imax] = max( abs(real(l(k:n)))+abs(imag(l(k:n))) ); */
		len = n-k;
		imax = izamax( &len, (DP)(l+k), &ione) - 1;
		/* if lmax == 0
			warning('drsolve:clsolve', ...
			'Matrix is singular to working precision.')
		end */
		if( (creal(l[k+imax]) == 0.) && (cimag(l[k+imax]) == 0.) )
			mexWarnMsgIdAndTxt( "drsolve:clsolve:singularMatrix",
				"Matrix is singular to working precision.");
		/* if imax > 1
			imax  = imax+(k-1);
			l([k imax])   = l([imax k]);
			t([k imax])   = t([imax k]);
			G([k imax],:) = G([imax k],:);
			B([k imax],:) = B([imax k],:);
			p([k imax])   = p([imax k]);
		   end
		   pivot = l(k); */
		if( imax > 0 ) {
			imax += k;
			pivot = l[imax];  l[imax] = l[k];
			scal = t[k];  t[k] = t[imax];  t[imax] = scal;
			zswap( &dr, (DP)(g+k), &n, (DP)(g+imax), &n);
			zswap( &kb, (DP)(b+k), &n, (DP)(b+imax), &n);
			i = p[k];  p[k] = p[imax];  p[imax] = i;
		} else
			pivot = l[k];
		/* l(k) = -1; */
		l[k] = minusone;
		/* u1(1:n-k) = (G(k,:)*H(:,k+1:n)) ./ (t(k)-s(k+1:n).'); */
		len = n-k-1;
		zgemv( notr, &len, &dr, (DP)&one, (DP)(h+k+1), &n, (DP)(g+k),
			&n, (DP)&zero, (DP)u1, &ione);
		zcopy( &len, (DP)(t+k), &izero, (DP)temp, &ione);
		zaxpy( &len, (DP)&minusone, (DP)(s+k+1), &ione, (DP)temp,
			&ione);
		ztbsv( lotr, notr, nut, &len, &izero, (DP)temp, &ione, (DP)u1,
			&ione);
		/* g1_pivot = G(k,:)/pivot; */
		zcopy( &dr, (DP)(g+k), &n, (DP)temp, &ione);
		/* G(k,:) = 0; */
		zcopy( &dr, (DP)&zero, &izero, (DP)(g+k), &n);
		/* G = G-l*g1_pivot; */
		scal = minusone / pivot;
		zgeru( &n, &dr, (DP)&scal, (DP)l, &ione, (DP)temp, &ione,
			(DP)g, &n);
		/* g2_pivot = B(k,:)/pivot; */
		zcopy( &kb, (DP)(b+k), &n, (DP)temp, &ione);
		/* B(k,:) = 0; */
		zcopy( &kb, (DP)&zero, &izero, (DP)(b+k), &n);
		/* B = B-l*g2_pivot; */
		zgeru( &n, &kb, (DP)&scal, (DP)l, &ione, (DP)temp, &ione,
			(DP)b, &n);
		/* h1_pivot = H(:,k)/pivot; */
		zcopy( &dr, (DP)(h+k), &n, (DP)temp, &ione);
		/* H(:,k+1:n) = H(:,k+1:n) - h1_pivot*u1(1:n-k); */
		len = n-k-1;
		zgeru( &len, &dr, (DP)&scal, (DP)u1, &ione, (DP)temp, &ione,
			(DP)(h+k+1), &n);
		/* uinvnm = max([uinvnm; ...
		   sum(abs(real(l(1:k)))+abs(imag(l(1:k)))) / abs(pivot) ]); */
		len = k+1;
		uinvnm = max( uinvnm, dzasum( &len, (DP)l, &ione)/fabs(pivot));
		/* ucolsum(k:n) = ucolsum(k:n) + ...
			[ abs(real(pivot))+abs(imag(pivot)), ...
			abs(real(u1(1:n-k)))+abs(imag(u1(1:n-k))) ]; */
		ucolsum[k] += fabs(creal(pivot))+fabs(cimag(pivot));
		for( i=0; i<n-k-1; i++)
			ucolsum[k+i+1] += fabs(creal(u1[i]))+fabs(cimag(u1[i]));
	}

	/* unorm = max(ucolsum); */
	k = idamax( &n, ucolsum, &ione) - 1;
	/* rcond = 1/unorm/uinvnm; */
	*rcond = one/ucolsum[k]/uinvnm;

	/* B(q,:) = B; */
	for( j=0; j<kb; j++) {
		for( i=0; i<n; i++)
			temp[q[i]-1] = b[i+n*j];
		zcopy( &n, (DP)temp, &ione, (DP)(b+n*j), &ione);
	}

	mxFree(temp);
	mxFree(tau);
	mxFree(ucolsum);
	mxFree(u1);
	mxFree(l);
}


/* complete pivoting - complex */
void zschurctp( PINT n, PINT dr, PINT kb, complex *g, complex *h, complex *t,
	complex *s, complex *b, double *rcond, mwSize *p, mwSize *q)
{
	PINT imax, jmax, k, i, j, len;
	complex *l, *u1, *Mj, *temp, pivot, scal;
	double uinvnm=0., *ucolsum, apivot;
	char notr[2] = {'N','\0'}, tran[2] = {'T','\0'}, cotr[2] = {'C','\0'};
	char lotr[2] = {'L','\0'}, nut[2] = {'N','\0'};
	complex one=1., minusone=-1., zero=0.;
	PINT ione=1, izero=0;

	l = mxCalloc( n, sizeof(complex));
	u1 = mxCalloc( n, sizeof(complex));
	ucolsum = mxCalloc( n, sizeof(double));
	Mj = mxCalloc( n, sizeof(complex));
	temp = mxCalloc( max(n,kb), sizeof(complex));

	for( k=0; k<n; k++) {
		/* for j=k:n
			Mj(k:n) = (G(k:n,:)*H(j,:)) ./ (t(k:n)-s(j));
			[ max_c(j), rv_ind(j) ] = max( abs(real(Mj(k:n))) + ...
						       abs(imag(Mj(k:n))) );
		end */
		len = n-k;
		apivot = 0.;
		imax = 0;  jmax = 0;
		for( j=k; j<n; j++) {
			zgemv( notr, &len, &dr, (DP)&one, (DP)(g+k), &n,
				(DP)(h+j), &n, (DP)&zero, (DP)Mj, &ione);
			zcopy( &len, (DP)(t+k), &ione, (DP)temp, &ione);
			zaxpy( &len, (DP)&minusone, (DP)(s+j), &izero,
				(DP)temp, &ione);
			ztbsv( lotr, notr, nut, &len, &izero, (DP)temp, &ione, 
				(DP)Mj, &ione);
			i = izamax( &len, (DP)Mj, &ione) - 1;
			if( fabs(creal(Mj[i]))+fabs(cimag(Mj[i])) > apivot ) {
				apivot = fabs(creal(Mj[i])) + fabs(cimag(Mj[i]));
				imax = i;
				jmax = j;
			}
		}
		imax = imax + k;
		/* if apivot == 0
			warning('drsolve:clsolve:singularMatrix', ...
                        'Matrix is singular to working precision.')
		end */
		if( apivot == 0. )
			mexWarnMsgIdAndTxt( "drsolve:clsolve:singularMatrix",
				"Matrix is singular to working precision.");
		/* if r_ind ~= k
			t([r_ind,k]  ) = t([k,r_ind]  );
			G([r_ind,k],:) = G([k,r_ind],:);
			B([r_ind,k],:) = B([k,r_ind],:);
			p([r_ind,k]  ) = p([k,r_ind]  );
		end */
		if( imax > k ) {
			scal = t[k];  t[k] = t[imax];  t[imax] = scal;
			zswap( &dr, (DP)(g+k), &n, (DP)(g+imax), &n);
			zswap( &kb, (DP)(b+k), &n, (DP)(b+imax), &n);
			i = p[k];  p[k] = p[imax];  p[imax] = i;
		}
		/* if c_ind ~= k
			s(  [c_ind,k]) = s(  [k,c_ind]);
			H(:,[c_ind,k]) = H(:,[k,c_ind]);
			q(  [c_ind,k]) = q(  [k,c_ind]);
		end */
		if( jmax > k ) {
			scal = s[k];  s[k] = s[jmax];  s[jmax] = scal;
			zswap( &dr, (DP)(h+k), &n, (DP)(h+jmax), &n);
			i = q[k];  q[k] = q[jmax];  q[jmax] = i;
		}
		/* l = (G*H(:,k)) ./ ([s(1:k-1);t(k:n)]-s(k)); */
		zgemv( notr, &n, &dr, (DP)&one, (DP)g, &n, (DP)(h+k), &n,
			(DP)&zero, (DP)l, &ione);
		len = k;
		zcopy( &len, (DP)s, &ione, (DP)temp, &ione);
		len = n-k;
		zcopy( &len, (DP)(t+k), &ione, (DP)(temp+k), &ione);
		zaxpy( &n, (DP)&minusone, (DP)(s+k), &izero, (DP)temp, &ione);
		ztbsv( lotr, notr, nut, &n, &izero, (DP)temp, &ione, (DP)l,
			&ione);
		/* pivot = l(k); */
		pivot = l[k];
		/* l(k) = -1; */
		l[k] = minusone;
		/* u1(1:n-k) = (G(k,:)*H(:,k+1:n)) ./ (t(k)-s(k+1:n).'); */
		len = n-k-1;
		zgemv( notr, &len, &dr, (DP)&one, (DP)(h+k+1), &n, (DP)(g+k),
			&n, (DP)&zero, (DP)u1, &ione);
		zcopy( &len, (DP)(t+k), &izero, (DP)temp, &ione);
		zaxpy( &len, (DP)&minusone, (DP)(s+k+1), &ione, (DP)temp,
			&ione);
		ztbsv( lotr, notr, nut, &len, &izero, (DP)temp, &ione, (DP)u1,
			&ione);
		/* g1_pivot = G(k,:)/pivot; */
		zcopy( &dr, (DP)(g+k), &n, (DP)temp, &ione);
		/* G(k,:) = 0; */
		zcopy( &dr, (DP)&zero, &izero, (DP)(g+k), &n);
		/* G = G-l*g1_pivot; */
		scal = minusone / pivot;
		zgeru( &n, &dr, (DP)&scal, (DP)l, &ione, (DP)temp, &ione,
			(DP)g, &n);
		/* g2_pivot = B(k,:)/pivot; */
		zcopy( &kb, (DP)(b+k), &n, (DP)temp, &ione);
		/* B(k,:) = 0; */
		zcopy( &kb, (DP)&zero, &izero, (DP)(b+k), &n);
		/* B = B-l*g2_pivot; */
		zgeru( &n, &kb, (DP)&scal, (DP)l, &ione, (DP)temp, &ione,
			(DP)b, &n);
		/* h1_pivot = H(:,k)/pivot; */
		zcopy( &dr, (DP)(h+k), &n, (DP)temp, &ione);
		/* H(:,k+1:n) = H(:,k+1:n) - h1_pivot*u1(1:n-k); */
		len = n-k-1;
		zgeru( &len, &dr, (DP)&scal, (DP)u1, &ione, (DP)temp, &ione,
			(DP)(h+k+1), &n);
		/* uinvnm = max([uinvnm; ...
		   sum(abs(real(l(1:k)))+abs(imag(l(1:k)))) / abs(pivot) ]); */
		len = k+1;
		uinvnm = max( uinvnm, dzasum( &len, (DP)l, &ione)/fabs(pivot));
		/* ucolsum(k:n) = ucolsum(k:n) + ...
			[ abs(real(pivot))+abs(imag(pivot)), ...
			abs(real(u1(1:n-k)))+abs(imag(u1(1:n-k))) ]; */
		ucolsum[k] += fabs(creal(pivot))+fabs(cimag(pivot));
		for( i=0; i<n-k-1; i++)
			ucolsum[k+i+1] += fabs(creal(u1[i]))+fabs(cimag(u1[i]));
	}

	/* unorm = max(ucolsum); */
	k = idamax( &n, ucolsum, &ione) - 1;
	/* rcond = 1/unorm/uinvnm; */
	*rcond = one/ucolsum[k]/uinvnm;

	/* B(q,:) = B; */
	for( j=0; j<kb; j++) {
		for( i=0; i<n; i++)
			temp[q[i]-1] = b[i+n*j];
		zcopy( &n, (DP)temp, &ione, (DP)(b+n*j), &ione);
	}

	mxFree(temp);
	mxFree(Mj);
	mxFree(ucolsum);
	mxFree(u1);
	mxFree(l);
}


/* extended Gu's pivoting - complex */
void zschurcguex( PINT n, PINT dr, PINT kb, PINT jstep, complex *g, complex *h,
	complex *t, complex *s, complex *b, double *rcond, mwSize *p, mwSize *q)
{
	PINT imax, k, i, j, len, ltemp, info;
	complex *l, *u1, *tau, *temp, pivot, scal;
	double uinvnm=0., *ucolsum, nmax, tt;
	char notr[2] = {'N','\0'}, tran[2] = {'T','\0'}, cotr[2] = {'C','\0'};
	char lotr[2] = {'L','\0'}, nut[2] = {'N','\0'};
	char uptr[2] = {'U','\0'}, right[2] = {'R','\0'};
	complex one=1., minusone=-1., zero=0.;
	PINT ione=1, izero=0;

	if( !jstep )
		jstep = 10;	/* default value */

	l = mxCalloc( n, sizeof(complex));
	u1 = mxCalloc( n, sizeof(complex));
	ucolsum = mxCalloc( n, sizeof(double));
	tau = mxCalloc( dr, sizeof(complex));

	/* get optimal dimension of work array */
	ltemp = -1;
	zgeqrf( &n, &dr, (DP)g, &n, (DP)tau, (DP)&scal, &ltemp, &info);
	if( info )  mexErrMsgTxt( "Error in zschurcguex/zgeqrf(init).");
	ltemp = (PINT) scal;
	ltemp = max(ltemp,n);
	ltemp = max(ltemp,kb);
	temp = mxCalloc( ltemp, sizeof(complex));

	for( k=0; k<n; k++) {
		/* if mod(k,Jmax_step) == 1
			[ G R ]    = qr(G,0);
			H(:,k:n) = R*H(:,k:n);
			[ nmax c_ind ] = max(sum( abs(real(H(:,k:end)))...
						+ abs(imag(H(:,k:end))), 1));
			c_ind = c_ind+(k-1);
			if c_ind ~= k
				s(  [c_ind,k]) = s(  [k,c_ind]);
				H(:,[c_ind,k]) = H(:,[k,c_ind]);
				q(  [c_ind,k]) = q(  [k,c_ind]);
			end
		end */
		if( (k % jstep) == 0 ) {
			zgeqrf( &n, &dr, (DP)g, &n, (DP)tau, (DP)temp, &ltemp,
				&info);
			if( info )  mexErrMsgTxt( "Error in zschurcguex/zgeqrf.");
			len = n-k;
			ztrmm( right, uptr, tran, nut, &len, &dr, (DP)&one,
				(DP)g, &n, (DP)(h+k), &n);
			zungqr( &n, &dr, &dr, (DP)g, &n, (DP)tau, (DP)temp,
				&ltemp, &info);
			if( info )  mexErrMsgTxt( "Error in zschurcguex/zungqr.");
			nmax = dzasum( &dr, (DP)(h+k), &n);
			j = k;
			for( i=k+1; i<n; i++) {
				tt = dzasum( &dr, (DP)(h+i), &n);
				if( tt > nmax ) {
					nmax = tt;
					j = i;
				}
			}
			if( j > k ) {
				scal = s[k];  s[k] = s[j];  s[j] = scal;
				zswap( &dr, (DP)(h+k), &n, (DP)(h+j), &n);
				i = q[k];  q[k] = q[j];  q[j] = i;
			}
		}
		/* l = (G*H(:,k)) ./ ([s(1:k-1);t(k:n)]-s(k)); */
		zgemv( notr, &n, &dr, (DP)&one, (DP)g, &n, (DP)(h+k), &n,
			(DP)&zero, (DP)l, &ione);
		len = k;
		zcopy( &len, (DP)s, &ione, (DP)temp, &ione);
		len = n-k;
		zcopy( &len, (DP)(t+k), &ione, (DP)(temp+k), &ione);
		zaxpy( &n, (DP)&minusone, (DP)(s+k), &izero, (DP)temp, &ione);
		ztbsv( lotr, notr, nut, &n, &izero, (DP)temp, &ione, (DP)l,
			&ione);
		/* [lmax imax] = max( abs(real(l(k:n)))+abs(imag(l(k:n))) ); */
		len = n-k;
		imax = izamax( &len, (DP)(l+k), &ione) - 1;
		/* if lmax == 0
			warning('drsolve:clsolve', ...
			'Matrix is singular to working precision.')
		end */
		if( (creal(l[k+imax]) == 0.) && (cimag(l[k+imax]) == 0.) )
			mexWarnMsgIdAndTxt( "drsolve:clsolve:singularMatrix",
				"Matrix is singular to working precision.");
		/* if imax > 1
			imax  = imax+(k-1);
			l([k imax])   = l([imax k]);
			t([k imax])   = t([imax k]);
			G([k imax],:) = G([imax k],:);
			B([k imax],:) = B([imax k],:);
			p([k imax])   = p([imax k]);
		   end
		   pivot = l(k); */
		if( imax > 0 ) {
			imax += k;
			pivot = l[imax];  l[imax] = l[k];
			scal = t[k];  t[k] = t[imax];  t[imax] = scal;
			zswap( &dr, (DP)(g+k), &n, (DP)(g+imax), &n);
			zswap( &kb, (DP)(b+k), &n, (DP)(b+imax), &n);
			i = p[k];  p[k] = p[imax];  p[imax] = i;
		} else
			pivot = l[k];
		/* l(k) = -1; */
		l[k] = minusone;
		/* u1(1:n-k) = (G(k,:)*H(:,k+1:n)) ./ (t(k)-s(k+1:n).'); */
		len = n-k-1;
		zgemv( notr, &len, &dr, (DP)&one, (DP)(h+k+1), &n, (DP)(g+k),
			&n, (DP)&zero, (DP)u1, &ione);
		zcopy( &len, (DP)(t+k), &izero, (DP)temp, &ione);
		zaxpy( &len, (DP)&minusone, (DP)(s+k+1), &ione, (DP)temp,
			&ione);
		ztbsv( lotr, notr, nut, &len, &izero, (DP)temp, &ione, (DP)u1,
			&ione);
		/* g1_pivot = G(k,:)/pivot; */
		zcopy( &dr, (DP)(g+k), &n, (DP)temp, &ione);
		/* G(k,:) = 0; */
		zcopy( &dr, (DP)&zero, &izero, (DP)(g+k), &n);
		/* G = G-l*g1_pivot; */
		scal = minusone / pivot;
		zgeru( &n, &dr, (DP)&scal, (DP)l, &ione, (DP)temp, &ione,
			(DP)g, &n);
		/* g2_pivot = B(k,:)/pivot; */
		zcopy( &kb, (DP)(b+k), &n, (DP)temp, &ione);
		/* B(k,:) = 0; */
		zcopy( &kb, (DP)&zero, &izero, (DP)(b+k), &n);
		/* B = B-l*g2_pivot; */
		zgeru( &n, &kb, (DP)&scal, (DP)l, &ione, (DP)temp, &ione,
			(DP)b, &n);
		/* h1_pivot = H(:,k)/pivot; */
		zcopy( &dr, (DP)(h+k), &n, (DP)temp, &ione);
		/* H(:,k+1:n) = H(:,k+1:n) - h1_pivot*u1(1:n-k); */
		len = n-k-1;
		zgeru( &len, &dr, (DP)&scal, (DP)u1, &ione, (DP)temp, &ione,
			(DP)(h+k+1), &n);
		/* uinvnm = max([uinvnm; ...
		   sum(abs(real(l(1:k)))+abs(imag(l(1:k)))) / abs(pivot) ]); */
		len = k+1;
		uinvnm = max( uinvnm, dzasum( &len, (DP)l, &ione)/fabs(pivot));
		/* ucolsum(k:n) = ucolsum(k:n) + ...
			[ abs(real(pivot))+abs(imag(pivot)), ...
			abs(real(u1(1:n-k)))+abs(imag(u1(1:n-k))) ]; */
		ucolsum[k] += fabs(creal(pivot))+fabs(cimag(pivot));
		for( i=0; i<n-k-1; i++)
			ucolsum[k+i+1] += fabs(creal(u1[i]))+fabs(cimag(u1[i]));
	}

	/* unorm = max(ucolsum); */
	k = idamax( &n, ucolsum, &ione) - 1;
	/* rcond = 1/unorm/uinvnm; */
	*rcond = one/ucolsum[k]/uinvnm;

	/* B(q,:) = B; */
	for( j=0; j<kb; j++) {
		for( i=0; i<n; i++)
			temp[q[i]-1] = b[i+n*j];
		zcopy( &n, (DP)temp, &ione, (DP)(b+n*j), &ione);
	}

	mxFree(temp);
	mxFree(tau);
	mxFree(ucolsum);
	mxFree(u1);
	mxFree(l);
}


/* no pivoting - double */
void dschurcnp( PINT n, PINT dr, PINT kb, double *g, double *h, double *t,
	double *s, double *b, double *rcond)
{
	PINT k, i, len;
	double *l, *u1, *temp, pivot, scal;
	double uinvnm=0., *ucolsum;
	char notr[2] = {'N','\0'}, tran[2] = {'T','\0'}, cotr[2] = {'C','\0'};
	char lotr[2] = {'L','\0'}, nut[2] = {'N','\0'};
	double one=1., minusone=-1., zero=0.;
	PINT ione=1, izero=0;

	l = mxCalloc( n, sizeof(double));
	u1 = mxCalloc( n, sizeof(double));
	ucolsum = mxCalloc( n, sizeof(double));
	temp = mxCalloc( max(n,kb), sizeof(double));

	for( k=0; k<n; k++) {
		/* l = (G*H(:,k)) ./ ([s(1:k-1);t(k:n)]-s(k)); */
                dgemv( notr, &n, &dr, &one, g, &n, h+k, &n, &zero, l, &ione);
		len = k;
		dcopy( &len, s, &ione, temp, &ione);
		len = n-k;
		dcopy( &len, t+k, &ione, temp+k, &ione);
		daxpy( &n, &minusone, s+k, &izero, temp, &ione);
		dtbsv( lotr, notr, nut, &n, &izero, temp, &ione, l, &ione);
		/* pivot = l(k); */
		pivot = l[k];
		/* l(k)  = -1; */
		l[k] = minusone;
		if( pivot == 0. )
			mexWarnMsgIdAndTxt(
				"drsolve:clsolve:nullDiagonalElement",
			"Null diagonal element. Try to activate pivoting.");
		/* u1(1:n-k) = (G(k,:)*H(:,k+1:n)) ./ (t(k)-s(k+1:n).'); */
		len = n-k-1;
		dgemv( notr, &len, &dr, &one, h+k+1, &n, g+k, &n,
			&zero, u1, &ione);
		dcopy( &len, t+k, &izero, temp, &ione);
		daxpy( &len, &minusone, s+k+1, &ione, temp, &ione);
		dtbsv( lotr, notr, nut, &len, &izero, temp, &ione, u1, &ione);
		/* g1_pivot = G(k,:)/pivot; */
		dcopy( &dr, g+k, &n, temp, &ione);
		/* G(k,:) = 0; */
		dcopy( &dr, &zero, &izero, g+k, &n);
		/* G = G-l*g1_pivot; */
		scal = minusone / pivot;
		dger( &n, &dr, &scal, l, &ione, temp, &ione, g, &n);
		/* g2_pivot = B(k,:)/pivot; */
		dcopy( &kb, b+k, &n, temp, &ione);
		/* B(k,:) = 0; */
		dcopy( &kb, &zero, &izero, b+k, &n);
		/* B = B-l*g2_pivot; */
		dger( &n, &kb, &scal, l, &ione, temp, &ione, b, &n);
		/* h1_pivot = H(:,k)/pivot; */
		dcopy( &dr, h+k, &n, temp, &ione);
		/* H(:,k+1:n) = H(:,k+1:n) - h1_pivot*u1(1:n-k); */
		len = n-k-1;
		dger( &len, &dr, &scal, u1, &ione, temp, &ione, h+k+1, &n);
		/* uinvnm = max([uinvnm; ...
		   sum(abs(real(l(1:k)))+abs(imag(l(1:k)))) / abs(pivot) ]); */
		len = k+1;
		uinvnm = max( uinvnm, dasum( &len, l, &ione)/fabs(pivot));
		/* ucolsum(k:n) = ucolsum(k:n) + ...
			[ abs(real(pivot))+abs(imag(pivot)), ...
			abs(real(u1(1:n-k)))+abs(imag(u1(1:n-k))) ]; */
		ucolsum[k] += fabs(pivot);
		for( i=0; i<n-k-1; i++)
			ucolsum[k+i+1] += fabs(u1[i]);
	}

	/* unorm = max(ucolsum); */
	k = idamax( &n, ucolsum, &ione) - 1;
	/* rcond = 1/unorm/uinvnm; */
	*rcond = one/ucolsum[k]/uinvnm;

	mxFree(temp);
	mxFree(ucolsum);
	mxFree(u1);
	mxFree(l);
}


/* partial pivoting - double */
void dschurcpp( PINT n, PINT dr, PINT kb, double *g, double *h, double *t,
	double *s, double *b, double *rcond, mwSize *p)
{
	PINT imax, k, i, len;
	double *l, *u1, *temp, pivot, scal;
	double uinvnm=0., *ucolsum;
	char notr[2] = {'N','\0'}, tran[2] = {'T','\0'}, cotr[2] = {'C','\0'};
	char lotr[2] = {'L','\0'}, nut[2] = {'N','\0'};
	double one=1., minusone=-1., zero=0.;
	PINT ione=1, izero=0;

	l = mxCalloc( n, sizeof(double));
	u1 = mxCalloc( n, sizeof(double));
	ucolsum = mxCalloc( n, sizeof(double));
	temp = mxCalloc( max(n,kb), sizeof(double));

	for( k=0; k<n; k++) {
		/* l = (G*H(:,k)) ./ ([s(1:k-1);t(k:n)]-s(k)); */
                dgemv( notr, &n, &dr, &one, g, &n, h+k, &n, &zero, l, &ione);
		len = k;
		dcopy( &len, s, &ione, temp, &ione);
		len = n-k;
		dcopy( &len, t+k, &ione, temp+k, &ione);
		daxpy( &n, &minusone, s+k, &izero, temp, &ione);
		dtbsv( lotr, notr, nut, &n, &izero, temp, &ione, l, &ione);
		/* [lmax imax] = max( abs(real(l(k:n)))+abs(imag(l(k:n))) ); */
		len = n-k;
		imax = idamax( &len, l+k, &ione) - 1;
		/* if lmax == 0
			warning('drsolve:clsolve', ...
			'Matrix is singular to working precision.')
		end */
		if( l[k+imax] == 0. )
			mexWarnMsgIdAndTxt( "drsolve:clsolve:singularMatrix",
				"Matrix is singular to working precision.");
		/* if imax > 1
			imax  = imax+(k-1);
			l([k imax])   = l([imax k]);
			t([k imax])   = t([imax k]);
			G([k imax],:) = G([imax k],:);
			B([k imax],:) = B([imax k],:);
			p([k imax])   = p([imax k]);
		    end
		    pivot = l(k); */
		if( imax > 0 ) {
			imax += k;
			pivot = l[imax];  l[imax] = l[k];
			scal = t[k];  t[k] = t[imax];  t[imax] = scal;
			dswap( &dr, g+k, &n, g+imax, &n);
			dswap( &kb, b+k, &n, b+imax, &n);
			i = p[k];  p[k] = p[imax];  p[imax] = i;
		} else
			pivot = l[k];
		/* l(k) = -1; */
		l[k] = minusone;
		/* u1(1:n-k) = (G(k,:)*H(:,k+1:n)) ./ (t(k)-s(k+1:n).'); */
		len = n-k-1;
		dgemv( notr, &len, &dr, &one, h+k+1, &n, g+k, &n, &zero, u1,
			&ione);
		dcopy( &len, t+k, &izero, temp, &ione);
		daxpy( &len, &minusone, s+k+1, &ione, temp, &ione);
		dtbsv( lotr, notr, nut, &len, &izero, temp, &ione, u1, &ione);
		/* g1_pivot = G(k,:)/pivot; */
		dcopy( &dr, g+k, &n, temp, &ione);
		/* G(k,:) = 0; */
		dcopy( &dr, &zero, &izero, g+k, &n);
		/* G = G-l*g1_pivot; */
		scal = minusone / pivot;
		dger( &n, &dr, &scal, l, &ione, temp, &ione, g, &n);
		/* g2_pivot = B(k,:)/pivot; */
		dcopy( &kb, b+k, &n, temp, &ione);
		/* B(k,:) = 0; */
		dcopy( &kb, &zero, &izero, b+k, &n);
		/* B = B-l*g2_pivot; */
		dger( &n, &kb, &scal, l, &ione, temp, &ione, b, &n);
		/* h1_pivot = H(:,k)/pivot; */
		dcopy( &dr, h+k, &n, temp, &ione);
		/* H(:,k+1:n) = H(:,k+1:n) - h1_pivot*u1(1:n-k); */
		len = n-k-1;
		dger( &len, &dr, &scal, u1, &ione, temp, &ione, h+k+1, &n);
		/* uinvnm = max([uinvnm; ...
		   sum(abs(real(l(1:k)))+abs(imag(l(1:k)))) / abs(pivot) ]); */
		len = k+1;
		uinvnm = max( uinvnm, dasum( &len, l, &ione)/fabs(pivot));
		/* ucolsum(k:n) = ucolsum(k:n) + ...
			[ abs(real(pivot))+abs(imag(pivot)), ...
			abs(real(u1(1:n-k)))+abs(imag(u1(1:n-k))) ]; */
		ucolsum[k] += fabs(pivot);
		for( i=0; i<n-k-1; i++)
			ucolsum[k+i+1] += fabs(u1[i]);
	}

	/* unorm = max(ucolsum); */
	k = idamax( &n, ucolsum, &ione) - 1;
	/* rcond = 1/unorm/uinvnm; */
	*rcond = one/ucolsum[k]/uinvnm;

	mxFree(temp);
	mxFree(ucolsum);
	mxFree(u1);
	mxFree(l);
}


/* partial pivoting with multiple s's - double */
void dschurcpps( PINT n, PINT dr, PINT kb, double *g, double *h, double *t,
	double *s, double *b, double *rcond, mwSize *p, mwSize *q)
{
	PINT imax=0, k, i, j, len, *i1, *i2;
	double *l, *u1, *temp, pivot, scal;
	double uinvnm=0., *ucolsum;
	char notr[2] = {'N','\0'}, tran[2] = {'T','\0'}, cotr[2] = {'C','\0'};
	char lotr[2] = {'L','\0'}, nut[2] = {'N','\0'};
	double one=1., minusone=-1., zero=0.;
	PINT ione=1, izero=0;

	l = mxCalloc( n, sizeof(double));
	u1 = mxCalloc( n, sizeof(double));
	ucolsum = mxCalloc( n, sizeof(double));
	temp = mxCalloc( max(n,kb), sizeof(double));
	i1 = mxCalloc( n, sizeof(PINT));
	i2 = mxCalloc( n, sizeof(PINT));

	/* [ss, a, b] = unique(s,'first');
	[ss, c, d] = unique(s,'last');
	i1 = a(b);
	i2 = c(d);
	delta = max(diff([a; numel(s)+1])); */
	i1[0] = 0;  scal = s[0];
	for( i=1; i<n; i++)
		if( scal == s[i] )
			i1[i] = i1[i-1];
		else {
			len = i-1;
			for( k=i1[len]; k<i; k++)
				i2[k] = len;
			i1[i] = i;  scal = s[i];
			imax = max( imax, i-i1[len]);
		}
	len = n-1;
	for( k=i1[len]; k<n; k++)
		i2[k] = len;
	imax = max( imax, n-i1[len]);
	/* if delta > dr
                 error('drsolve:clsolve:singularMatrix', ...
                        'Matrix is singular to working precision.')
	end */
	if( imax > dr )
		mexErrMsgIdAndTxt( "drsolve:clsolve:singularMatrix",
			"Matrix is singular.");

	for( k=0; k<n; k++) {
		/* l(k:n) = (G(k:n,:) *H(:,k)) ./ (t(k:n) -s(k)); */
		len = n-k;
                dgemv( notr, &len, &dr, &one, g+k, &n, h+k, &n, &zero, 
			l+k, &ione);
		dcopy( &len, t+k, &ione, temp+k, &ione);
		daxpy( &len, &minusone, s+k, &izero, temp+k, &ione);
		dtbsv( lotr, notr, nut, &len, &izero, temp+k, &ione, 
			l+k, &ione);
		/* l(1:i1(k)-1) = (G(1:i1(k)-1,:)*H(:,k)) ./ 
		  				(s(1:i1(k)-1)-s(k)); */
		len = i1[k];
		if( len ) {
			dgemv( notr, &len, &dr, &one, g, &n, h+k, &n, &zero, 
				l, &ione);
			dcopy( &len, s, &ione, temp, &ione);
			daxpy( &len, &minusone, s+k, &izero, temp, &ione);
			dtbsv( lotr, notr, nut, &len, &izero, temp, &ione, 
				l, &ione);
		}
		/* if i1(k) < k
			l(i1(k):k-1) = H(k-i1(k),i1(k):k-1).';
		   end */
		if( i1[k] < k ) {
			len = k-i1[k];
			dcopy( &len, h+(len-1)*n+i1[k], &ione, l+i1[k], &ione);
		}
		/* [lmax imax] = max( abs(real(l(k:n)))+abs(imag(l(k:n))) ); */
		len = n-k;
		imax = idamax( &len, l+k, &ione) - 1;
		/* if lmax == 0
			warning('drsolve:clsolve', ...
			'Matrix is singular to working precision.')
		   end */
		if( l[k+imax] == 0. )
			mexWarnMsgIdAndTxt( "drsolve:clsolve:singularMatrix",
				"Matrix is singular to working precision.");
		/* if imax > 1
			imax  = imax+(k-1);
			l([k imax])   = l([imax k]);
			t([k imax])   = t([imax k]);
			G([k imax],:) = G([imax k],:);
			B([k imax],:) = B([imax k],:);
			p([k imax])   = p([imax k]);
		   end
		   pivot = l(k); */
		if( imax > 0 ) {
			imax += k;
			pivot = l[imax];  l[imax] = l[k];
			scal = t[k];  t[k] = t[imax];  t[imax] = scal;
			dswap( &dr, g+k, &n, g+imax, &n);
			dswap( &kb, b+k, &n, b+imax, &n);
			i = p[k];  p[k] = p[imax];  p[imax] = i;
		} else
			pivot = l[k];
		/* l(k) = -1; */
		l[k] = minusone;
		/* u1(1:n-k) = (G(k,:)*H(:,k+1:n)) ./ (t(k)-s(k+1:n).'); */
		len = n-k-1;
		dgemv( notr, &len, &dr, &one, h+k+1, &n, g+k, &n, &zero, 
			u1, &ione);
		dcopy( &len, t+k, &izero, temp, &ione);
		daxpy( &len, &minusone, s+k+1, &ione, temp, &ione);
		dtbsv( lotr, notr, nut, &len, &izero, temp, &ione, u1, &ione);
		/* g1_pivot = G(k,:)/pivot; */
		dcopy( &dr, g+k, &n, temp, &ione);
		/* G(k,:) = 0; */
		dcopy( &dr, &zero, &izero, g+k, &n);
		/* G = G-l*g1_pivot; */
		scal = minusone / pivot;
		dger( &n, &dr, &scal, l, &ione, temp, &ione, g, &n);
		/* g2_pivot = B(k,:)/pivot; */
		dcopy( &kb, b+k, &n, temp, &ione);
		/* B(k,:) = 0; */
		dcopy( &kb, &zero, &izero, b+k, &n);
		/* B = B-l*g2_pivot; */
		dger( &n, &kb, &scal, l, &ione, temp, &ione, b, &n);
		/* h1_pivot = H(:,k)/pivot; */
		dcopy( &dr, h+k, &n, temp, &ione);
		/* H(:,k+1:n) = H(:,k+1:n) - h1_pivot*u1(1:n-k); */
		len = n-k-1;
		dger( &len, &dr, &scal, u1, &ione, temp, &ione, h+k+1, &n);
		/* if k < i2(k)
			pos = k-i1(k)+1;
			len = i2(k)-i1(k);
			H(pos:len,k)       = 0;
			H(pos:len,i1(k):k) = H(pos:len,i1(k):k) - 
					     u1(1:i2(k)-k).'*(l(i1(k):k).'/
					     pivot);
		   end */
		if( k < i2[k] ) {
			j = k-i1[k]+1;
			len = i2[k]-k;
			dcopy( &len, &zero, &izero, h+(j-1)*n+k, &n);
			scal = minusone / pivot;
			dger( &j, &len, &scal, l+i1[k], &ione, u1, &ione, 
				h+(j-1)*n+i1[k], &n);
		}
		/* uinvnm = max([uinvnm; ...
		   sum(abs(real(l(1:k)))+abs(imag(l(1:k)))) / abs(pivot) ]); */
		len = k+1;
		uinvnm = max( uinvnm, dasum( &len, l, &ione)/fabs(pivot));
		/* ucolsum(k:n) = ucolsum(k:n) + ...
			[ abs(real(pivot))+abs(imag(pivot)), ...
			abs(real(u1(1:n-k)))+abs(imag(u1(1:n-k))) ]; */
		ucolsum[k] += fabs(pivot);
		for( i=0; i<n-k-1; i++)
			ucolsum[k+i+1] += fabs(u1[i]);
	}

	/* unorm = max(ucolsum); */
	k = idamax( &n, ucolsum, &ione) - 1;
	/* rcond = 1/unorm/uinvnm; */
	*rcond = one/ucolsum[k]/uinvnm;

	/* B(q,:) = B; */
	for( j=0; j<kb; j++) {
		for( i=0; i<n; i++)
			temp[q[i]-1] = b[i+n*j];
		dcopy( &n, temp, &ione, b+n*j, &ione);
	}

	mxFree(i2);
	mxFree(i1);
	mxFree(temp);
	mxFree(ucolsum);
	mxFree(u1);
	mxFree(l);
}


/* Sweet&Brent's pivoting - double */
void dschurcsb( PINT n, PINT dr, PINT kb, double *g, double *h, double *t,
	double *s, double *b, double *rcond, mwSize *p, mwSize *q)
{
	PINT imax, jmax, k, i, j, len;
	double *l, *u1, *temp, pivot, scal;
	double uinvnm=0., *ucolsum, p1, p2;
	char notr[2] = {'N','\0'}, tran[2] = {'T','\0'}, cotr[2] = {'C','\0'};
	char lotr[2] = {'L','\0'}, nut[2] = {'N','\0'};
	double one=1., minusone=-1., zero=0.;
	PINT ione=1, izero=0;

	l = mxCalloc( n, sizeof(double));
	u1 = mxCalloc( n, sizeof(double));
	ucolsum = mxCalloc( n, sizeof(double));
	temp = mxCalloc( max(n,kb), sizeof(double));

	for( k=0; k<n; k++) {
		/* l(k:n) = (G(k:n,:)*H(:,k)) ./ (t(k:n)-s(k)); */
		len = n-k;
                dgemv( notr, &len, &dr, &one, g+k, &n, h+k, &n, &zero, 
			l+k, &ione);
		dcopy( &len, t+k, &ione, temp, &ione);
		daxpy( &len, &minusone, s+k, &izero, temp, &ione);
		dtbsv( lotr, notr, nut, &len, &izero, temp, &ione, l+k, &ione);
		/* [p1,i1] = max(abs(real(l(k:n)))+abs(imag(l(k:n))));
		r_ind = i1+k-1; */
		imax = idamax( &len, l+k, &ione) - 1;
		imax += k;
		p1 = fabs(creal(l[imax])) + fabs(cimag(l[imax]));
		/* u1(1:n-k) = (G(k,:)*H(:,k+1:n)) ./ (t(k)-s(k+1:n).');
		[p2,i2] = max(abs(real(u1(1:n-k)))+abs(imag(u1(1:n-k))));
		c_ind = i2+k; */
		len = n-k-1;
		if( len ) {
			dgemv( notr, &len, &dr, &one, h+k+1, &n, g+k, &n,
				&zero, u1, &ione);
			dcopy( &len, t+k, &izero, temp, &ione);
			daxpy( &len, &minusone, s+k+1, &ione, temp, &ione);
			dtbsv( lotr, notr, nut, &len, &izero, temp, &ione, 
				u1, &ione);
			jmax = idamax( &len, u1, &ione) - 1;
			p2 = fabs(creal(u1[jmax])) + fabs(cimag(u1[jmax]));
			jmax = jmax+k+1;
		} else
			p2 = 0.;
		/* if ( max(p1,p2)==0 )
			error('drsolve:clsolve:singularMatrix', ...
			'Matrix is singular to working precision.') */
		if( max(p1,p2) == 0. )
			mexWarnMsgIdAndTxt( "drsolve:clsolve:singularMatrix",
				"Matrix is singular to working precision.");
		/* elseif ( p2>p1 )
			s(  [c_ind,k]) = s(  [k,c_ind]);
			H(:,[c_ind,k]) = H(:,[k,c_ind]);
			q(  [c_ind,k]) = q(  [k,c_ind]);
			%
			u1(i2) = l(k);                                 % update
			l(k:n) = ( G(k:n,:)*H(:,k) ) ./ (t(k:n)-s(k)); % recompute
		end */
		else if( p2>p1 ) {
			scal = s[k];  s[k] = s[jmax];  s[jmax] = scal;
			dswap( &dr, h+k, &n, h+jmax, &n);
			i = q[k];  q[k] = q[jmax];  q[jmax] = i;
			u1[jmax-k-1] = l[k];
			len = n-k;
			dgemv( notr, &len, &dr, &one, g+k, &n, h+k, &n, &zero, 
				l+k, &ione);
			dcopy( &len, t+k, &ione, temp, &ione);
			daxpy( &len, &minusone, s+k, &izero, temp, &ione);
			dtbsv( lotr, notr, nut, &len, &izero, temp, &ione, 
				l+k, &ione);
		} else {
		/* if r_ind ~= k
			l([r_ind,k]  ) = l([k,r_ind]  );
			t([r_ind,k]  ) = t([k,r_ind]  );
			G([r_ind,k],:) = G([k,r_ind],:);
			B([r_ind,k],:) = B([k,r_ind],:);
			p([r_ind,k]  ) = p([k,r_ind]  );
			u1(1:n-k) = (G(k,:)*H(:,k+1:n)) ./ (t(k)-s(k+1:n).');
		   end */
			if( imax > k ) {
				pivot = l[imax];  l[imax] = l[k];  l[k] = pivot;
				scal = t[k];  t[k] = t[imax];  t[imax] = scal;
				dswap( &dr, g+k, &n, g+imax, &n);
				dswap( &kb, b+k, &n, b+imax, &n);
				i = p[k];  p[k] = p[imax];  p[imax] = i;
				len = n-k-1;
				dgemv( notr, &len, &dr, &one, h+k+1, &n, g+k,
					&n, &zero, u1, &ione);
				dcopy( &len, t+k, &izero, temp, &ione);
				daxpy( &len, &minusone, s+k+1, &ione, temp,
					&ione);
				dtbsv( lotr, notr, nut, &len, &izero, temp,
					&ione, u1, &ione);
			}
		}
		/* l(1:k-1) = (G(1:k-1,:)*H(:,k)) ./ (s(1:k-1)-s(k)); */
		if( k ) {
			dgemv( notr, &k, &dr, &one, g, &n, h+k, &n, &zero, 
				l, &ione);
			dcopy( &k, s, &ione, temp, &ione);
			daxpy( &k, &minusone, s+k, &izero, temp, &ione);
			dtbsv( lotr, notr, nut, &k, &izero, temp, &ione, 
				l, &ione);
		}
		/* pivot = l(k);
		   l(k) = -1; */
		pivot = l[k];
		l[k] = minusone;
		/* g1_pivot = G(k,:)/pivot; */
		dcopy( &dr, g+k, &n, temp, &ione);
		/* G(k,:) = 0; */
		dcopy( &dr, &zero, &izero, g+k, &n);
		/* G = G-l*g1_pivot; */
		scal = minusone / pivot;
		dger( &n, &dr, &scal, l, &ione, temp, &ione, g, &n);
		/* g2_pivot = B(k,:)/pivot; */
		dcopy( &kb, b+k, &n, temp, &ione);
		/* B(k,:) = 0; */
		dcopy( &kb, &zero, &izero, b+k, &n);
		/* B = B-l*g2_pivot; */
		dger( &n, &kb, &scal, l, &ione, temp, &ione, b, &n);
		/* h1_pivot = H(:,k)/pivot; */
		dcopy( &dr, h+k, &n, temp, &ione);
		/* H(:,k+1:n) = H(:,k+1:n) - h1_pivot*u1(1:n-k); */
		len = n-k-1;
		if( len )
			dger( &len, &dr, &scal, u1, &ione, temp, &ione, 
				h+k+1, &n);
		/* uinvnm = max([uinvnm; ...
		   sum(abs(real(l(1:k)))+abs(imag(l(1:k)))) / abs(pivot) ]); */
		len = k+1;
		uinvnm = max( uinvnm, dasum( &len, l, &ione)/fabs(pivot));
		/* ucolsum(k:n) = ucolsum(k:n) + ...
			[ abs(real(pivot))+abs(imag(pivot)), ...
			abs(real(u1(1:n-k)))+abs(imag(u1(1:n-k))) ]; */
		ucolsum[k] += fabs(creal(pivot))+fabs(cimag(pivot));
		for( i=0; i<n-k-1; i++)
			ucolsum[k+i+1] += fabs(creal(u1[i]))+fabs(cimag(u1[i]));
	}

	/* unorm = max(ucolsum); */
	k = idamax( &n, ucolsum, &ione) - 1;
	/* rcond = 1/unorm/uinvnm; */
	*rcond = one/ucolsum[k]/uinvnm;

	/* B(q,:) = B; */
	for( j=0; j<kb; j++) {
		for( i=0; i<n; i++)
			temp[q[i]-1] = b[i+n*j];
		dcopy( &n, temp, &ione, b+n*j, &ione);
	}

	mxFree(temp);
	mxFree(ucolsum);
	mxFree(u1);
	mxFree(l);
}


/* Gu's pivoting - double */
void dschurcgu( PINT n, PINT dr, PINT kb, PINT jstep, double *g, double *h,
	double *t, double *s, double *b, double *rcond, mwSize *p, mwSize *q)
{
	PINT imax, k, i, j, len, ltemp, info;
	double *l, *u1, *tau, *temp, pivot, scal;
	double uinvnm=0., *ucolsum, nmax, tt;
	char notr[2] = {'N','\0'}, tran[2] = {'T','\0'}, cotr[2] = {'C','\0'};
	char lotr[2] = {'L','\0'}, nut[2] = {'N','\0'};
	char uptr[2] = {'U','\0'}, right[2] = {'R','\0'};
	double one=1., minusone=-1., zero=0.;
	PINT ione=1, izero=0;

	if( !jstep )
		jstep = 10;	/* default value */

	l = mxCalloc( n, sizeof(double));
	u1 = mxCalloc( n, sizeof(double));
	ucolsum = mxCalloc( n, sizeof(double));
	tau = mxCalloc( dr, sizeof(double));

	/* get optimal dimension of work array */
	ltemp = -1;
	dgeqrf( &n, &dr, g, &n, tau, &scal, &ltemp, &info);
	if( info )  mexErrMsgTxt( "Error in dschurcgu/dgeqrf(init).");
	ltemp = (PINT) scal;
	ltemp = max(ltemp,n);
	ltemp = max(ltemp,kb);
	temp = mxCalloc( ltemp, sizeof(double));

	for( k=0; k<n; k++) {
		/* if (mod(k,Jmax_step) == 1) && (k <= n-dr+1)
			[ G(k:n,:) R ] = qr(G(k:n,:),0);
			G(1:k-1,:) = G(1:k-1,:) / R;
			H(:,k:n) = R*H(:,k:n);
			[ nmax c_ind ] = max(sum( abs(real(H(:,k:end)))...
						+ abs(imag(H(:,k:end))), 1));
			c_ind = c_ind+(k-1);
			if c_ind ~= k
				s(  [c_ind,k]) = s(  [k,c_ind]);
				H(:,[c_ind,k]) = H(:,[k,c_ind]);
				q(  [c_ind,k]) = q(  [k,c_ind]);
			end
		end */
		if( ((k % jstep) == 0) && (k <= n-dr) ) {
			len = n-k;
			dgeqrf( &len, &dr, g+k, &n, tau, temp, &ltemp, &info);
			if( info )  mexErrMsgTxt( "Error in dschurcgu/dgeqrf.");
			dtrmm( right, uptr, tran, nut, &len, &dr, &one,
				g+k, &n, h+k, &n);
			dtrsm( right, uptr, notr, nut, &k, &dr, &one, 
				g+k, &n, g, &n);
			dorgqr( &len, &dr, &dr, g+k, &n, tau, temp, 
				&ltemp, &info);
			if( info )  mexErrMsgTxt( "Error in dschurcgu/dorgqr.");
			nmax = dasum( &dr, h+k, &n);
			j = k;
			for( i=k+1; i<n; i++) {
				tt = dasum( &dr, h+i, &n);
				if( tt > nmax ) {
					nmax = tt;
					j = i;
				}
			}
			if( j > k ) {
				scal = s[k];  s[k] = s[j];  s[j] = scal;
				dswap( &dr, h+k, &n, h+j, &n);
				i = q[k];  q[k] = q[j];  q[j] = i;
			}
		}
		/* l = (G*H(:,k)) ./ ([s(1:k-1);t(k:n)]-s(k)); */
                dgemv( notr, &n, &dr, &one, g, &n, h+k, &n, &zero, l, &ione);
		len = k;
		dcopy( &len, s, &ione, temp, &ione);
		len = n-k;
		dcopy( &len, t+k, &ione, temp+k, &ione);
		daxpy( &n, &minusone, s+k, &izero, temp, &ione);
		dtbsv( lotr, notr, nut, &n, &izero, temp, &ione, l, &ione);
		/* [lmax imax] = max( abs(real(l(k:n)))+abs(imag(l(k:n))) ); */
		len = n-k;
		imax = idamax( &len, l+k, &ione) - 1;
		/* if lmax == 0
			warning('drsolve:clsolve', ...
			'Matrix is singular to working precision.')
		end */
		if( (creal(l[k+imax]) == 0.) && (cimag(l[k+imax]) == 0.) )
			mexWarnMsgIdAndTxt( "drsolve:clsolve:singularMatrix",
				"Matrix is singular to working precision.");
		/* if imax > 1
			imax  = imax+(k-1);
			l([k imax])   = l([imax k]);
			t([k imax])   = t([imax k]);
			G([k imax],:) = G([imax k],:);
			B([k imax],:) = B([imax k],:);
			p([k imax])   = p([imax k]);
		   end
		   pivot = l(k); */
		if( imax > 0 ) {
			imax += k;
			pivot = l[imax];  l[imax] = l[k];
			scal = t[k];  t[k] = t[imax];  t[imax] = scal;
			dswap( &dr, g+k, &n, g+imax, &n);
			dswap( &kb, b+k, &n, b+imax, &n);
			i = p[k];  p[k] = p[imax];  p[imax] = i;
		} else
			pivot = l[k];
		/* l(k) = -1; */
		l[k] = minusone;
		/* u1(1:n-k) = (G(k,:)*H(:,k+1:n)) ./ (t(k)-s(k+1:n).'); */
		len = n-k-1;
		dgemv( notr, &len, &dr, &one, h+k+1, &n, g+k, &n,
			&zero, u1, &ione);
		dcopy( &len, t+k, &izero, temp, &ione);
		daxpy( &len, &minusone, s+k+1, &ione, temp, &ione);
		dtbsv( lotr, notr, nut, &len, &izero, temp, &ione, u1, &ione);
		/* g1_pivot = G(k,:)/pivot; */
		dcopy( &dr, g+k, &n, temp, &ione);
		/* G(k,:) = 0; */
		dcopy( &dr, &zero, &izero, g+k, &n);
		/* G = G-l*g1_pivot; */
		scal = minusone / pivot;
		dger( &n, &dr, &scal, l, &ione, temp, &ione, g, &n);
		/* g2_pivot = B(k,:)/pivot; */
		dcopy( &kb, b+k, &n, temp, &ione);
		/* B(k,:) = 0; */
		dcopy( &kb, &zero, &izero, b+k, &n);
		/* B = B-l*g2_pivot; */
		dger( &n, &kb, &scal, l, &ione, temp, &ione, b, &n);
		/* h1_pivot = H(:,k)/pivot; */
		dcopy( &dr, h+k, &n, temp, &ione);
		/* H(:,k+1:n) = H(:,k+1:n) - h1_pivot*u1(1:n-k); */
		len = n-k-1;
		dger( &len, &dr, &scal, u1, &ione, temp, &ione, h+k+1, &n);
		/* uinvnm = max([uinvnm; ...
		   sum(abs(real(l(1:k)))+abs(imag(l(1:k)))) / abs(pivot) ]); */
		len = k+1;
		uinvnm = max( uinvnm, dasum( &len, l, &ione)/fabs(pivot));
		/* ucolsum(k:n) = ucolsum(k:n) + ...
			[ abs(real(pivot))+abs(imag(pivot)), ...
			abs(real(u1(1:n-k)))+abs(imag(u1(1:n-k))) ]; */
		ucolsum[k] += fabs(creal(pivot))+fabs(cimag(pivot));
		for( i=0; i<n-k-1; i++)
			ucolsum[k+i+1] += fabs(creal(u1[i]))+fabs(cimag(u1[i]));
	}

	/* unorm = max(ucolsum); */
	k = idamax( &n, ucolsum, &ione) - 1;
	/* rcond = 1/unorm/uinvnm; */
	*rcond = one/ucolsum[k]/uinvnm;

	/* B(q,:) = B; */
	for( j=0; j<kb; j++) {
		for( i=0; i<n; i++)
			temp[q[i]-1] = b[i+n*j];
		dcopy( &n, temp, &ione, b+n*j, &ione);
	}

	mxFree(temp);
	mxFree(tau);
	mxFree(ucolsum);
	mxFree(u1);
	mxFree(l);
}


/* complete pivoting - double */
void dschurctp( PINT n, PINT dr, PINT kb, double *g, double *h, double *t,
	double *s, double *b, double *rcond, mwSize *p, mwSize *q)
{
	PINT imax, jmax, k, i, j, len;
	double *l, *u1, *Mj, *temp, pivot, scal;
	double uinvnm=0., *ucolsum, apivot;
	char notr[2] = {'N','\0'}, tran[2] = {'T','\0'}, cotr[2] = {'C','\0'};
	char lotr[2] = {'L','\0'}, nut[2] = {'N','\0'};
	double one=1., minusone=-1., zero=0.;
	PINT ione=1, izero=0;

	l = mxCalloc( n, sizeof(double));
	u1 = mxCalloc( n, sizeof(double));
	ucolsum = mxCalloc( n, sizeof(double));
	Mj = mxCalloc( n, sizeof(double));
	temp = mxCalloc( max(n,kb), sizeof(double));

	for( k=0; k<n; k++) {
		/* for j=k:n
			Mj(k:n) = (G(k:n,:)*H(j,:)) ./ (t(k:n)-s(j));
			[ max_c(j), rv_ind(j) ] = max( abs(real(Mj(k:n))) + ...
						       abs(imag(Mj(k:n))) );
		end */
		len = n-k;
		apivot = 0.;
		imax = 0;  jmax = 0;
		for( j=k; j<n; j++) {
			dgemv( notr, &len, &dr, &one, g+k, &n, h+j, &n,
				&zero, Mj, &ione);
			dcopy( &len, t+k, &ione, temp, &ione);
			daxpy( &len, &minusone, s+j, &izero, temp, &ione);
			dtbsv( lotr, notr, nut, &len, &izero, temp, &ione, 
				Mj, &ione);
			i = idamax( &len, Mj, &ione) - 1;
			if( fabs(Mj[i]) > apivot ) {
				apivot = fabs(Mj[i]);
				imax = i;
				jmax = j;
			}
		}
		imax = imax + k;
		/* if apivot == 0
			warning('drsolve:clsolve:singularMatrix', ...
                        'Matrix is singular to working precision.')
		end */
		if( apivot == 0. )
			mexWarnMsgIdAndTxt( "drsolve:clsolve:singularMatrix",
				"Matrix is singular to working precision.");
		/* if r_ind ~= k
			t([r_ind,k]  ) = t([k,r_ind]  );
			G([r_ind,k],:) = G([k,r_ind],:);
			B([r_ind,k],:) = B([k,r_ind],:);
			p([r_ind,k]  ) = p([k,r_ind]  );
		end */
		if( imax > k ) {
			scal = t[k];  t[k] = t[imax];  t[imax] = scal;
			dswap( &dr, g+k, &n, g+imax, &n);
			dswap( &kb, b+k, &n, b+imax, &n);
			i = p[k];  p[k] = p[imax];  p[imax] = i;
		}
		/* if c_ind ~= k
			s(  [c_ind,k]) = s(  [k,c_ind]);
			H(:,[c_ind,k]) = H(:,[k,c_ind]);
			q(  [c_ind,k]) = q(  [k,c_ind]);
		end */
		if( jmax > k ) {
			scal = s[k];  s[k] = s[jmax];  s[jmax] = scal;
			dswap( &dr, h+k, &n, h+jmax, &n);
			i = q[k];  q[k] = q[jmax];  q[jmax] = i;
		}
		/* l = (G*H(:,k)) ./ ([s(1:k-1);t(k:n)]-s(k)); */
                dgemv( notr, &n, &dr, &one, g, &n, h+k, &n, &zero, l, &ione);
		len = k;
		dcopy( &len, s, &ione, temp, &ione);
		len = n-k;
		dcopy( &len, t+k, &ione, temp+k, &ione);
		daxpy( &n, &minusone, s+k, &izero, temp, &ione);
		dtbsv( lotr, notr, nut, &n, &izero, temp, &ione, l, &ione);
		/* pivot = l(k); */
		pivot = l[k];
		/* l(k) = -1; */
		l[k] = minusone;
		/* u1(1:n-k) = (G(k,:)*H(:,k+1:n)) ./ (t(k)-s(k+1:n).'); */
		len = n-k-1;
		dgemv( notr, &len, &dr, &one, h+k+1, &n, g+k, &n,
			&zero, u1, &ione);
		dcopy( &len, t+k, &izero, temp, &ione);
		daxpy( &len, &minusone, s+k+1, &ione, temp, &ione);
		dtbsv( lotr, notr, nut, &len, &izero, temp, &ione, u1, &ione);
		/* g1_pivot = G(k,:)/pivot; */
		dcopy( &dr, g+k, &n, temp, &ione);
		/* G(k,:) = 0; */
		dcopy( &dr, &zero, &izero, g+k, &n);
		/* G = G-l*g1_pivot; */
		scal = minusone / pivot;
		dger( &n, &dr, &scal, l, &ione, temp, &ione, g, &n);
		/* g2_pivot = B(k,:)/pivot; */
		dcopy( &kb, b+k, &n, temp, &ione);
		/* B(k,:) = 0; */
		dcopy( &kb, &zero, &izero, b+k, &n);
		/* B = B-l*g2_pivot; */
		dger( &n, &kb, &scal, l, &ione, temp, &ione, b, &n);
		/* h1_pivot = H(:,k)/pivot; */
		dcopy( &dr, h+k, &n, temp, &ione);
		/* H(:,k+1:n) = H(:,k+1:n) - h1_pivot*u1(1:n-k); */
		len = n-k-1;
		dger( &len, &dr, &scal, u1, &ione, temp, &ione, h+k+1, &n);
		/* uinvnm = max([uinvnm; ...
		   sum(abs(real(l(1:k)))+abs(imag(l(1:k)))) / abs(pivot) ]); */
		len = k+1;
		uinvnm = max( uinvnm, dasum( &len, l, &ione)/fabs(pivot));
		/* ucolsum(k:n) = ucolsum(k:n) + ...
			[ abs(real(pivot))+abs(imag(pivot)), ...
			abs(real(u1(1:n-k)))+abs(imag(u1(1:n-k))) ]; */
		ucolsum[k] += fabs(pivot);
		for( i=0; i<n-k-1; i++)
			ucolsum[k+i+1] += fabs(u1[i]);
	}

	/* unorm = max(ucolsum); */
	k = idamax( &n, ucolsum, &ione) - 1;
	/* rcond = 1/unorm/uinvnm; */
	*rcond = one/ucolsum[k]/uinvnm;

	/* B(q,:) = B; */
	for( j=0; j<kb; j++) {
		for( i=0; i<n; i++)
			temp[q[i]-1] = b[i+n*j];
		dcopy( &n, temp, &ione, b+n*j, &ione);
	}

	mxFree(temp);
	mxFree(Mj);
	mxFree(ucolsum);
	mxFree(u1);
	mxFree(l);
}


/* extended Gu's pivoting - double */
void dschurcguex( PINT n, PINT dr, PINT kb, PINT jstep, double *g, double *h,
	double *t, double *s, double *b, double *rcond, mwSize *p, mwSize *q)
{
	PINT imax, k, i, j, len, ltemp, info;
	double *l, *u1, *tau, *temp, pivot, scal;
	double uinvnm=0., *ucolsum, nmax, tt;
	char notr[2] = {'N','\0'}, tran[2] = {'T','\0'}, cotr[2] = {'C','\0'};
	char lotr[2] = {'L','\0'}, nut[2] = {'N','\0'};
	char uptr[2] = {'U','\0'}, right[2] = {'R','\0'};
	double one=1., minusone=-1., zero=0.;
	PINT ione=1, izero=0;

	if( !jstep )
		jstep = 10;	/* default value */

	l = mxCalloc( n, sizeof(double));
	u1 = mxCalloc( n, sizeof(double));
	ucolsum = mxCalloc( n, sizeof(double));
	tau = mxCalloc( dr, sizeof(double));

	/* get optimal dimension of work array */
	ltemp = -1;
	dgeqrf( &n, &dr, g, &n, tau, &scal, &ltemp, &info);
	if( info )  mexErrMsgTxt( "Error in dschurcguex/dgeqrf(init).");
	ltemp = (PINT) scal;
	ltemp = max(ltemp,n);
	ltemp = max(ltemp,kb);
	temp = mxCalloc( ltemp, sizeof(double));

	for( k=0; k<n; k++) {
		/* if mod(k,Jmax_step) == 1
			[ G R ]    = qr(G,0);
			H(:,k:n) = R*H(:,k:n);
			[ nmax c_ind ] = max(sum( abs(real(H(:,k:end)))...
						+ abs(imag(H(:,k:end))), 1));
			c_ind = c_ind+(k-1);
			if c_ind ~= k
				s(  [c_ind,k]) = s(  [k,c_ind]);
				H(:,[c_ind,k]) = H(:,[k,c_ind]);
				q(  [c_ind,k]) = q(  [k,c_ind]);
			end
		end */
		if( (k % jstep) == 1 ) {
			dgeqrf( &n, &dr, g, &n, tau, temp, &ltemp, &info);
			if( info )  mexErrMsgTxt( "Error in dschurcguex/dgeqrf.");
			len = n-k;
			dtrmm( right, uptr, tran, nut, &len, &dr, &one,
				g, &n, h+k, &n);
			dorgqr( &n, &dr, &dr, g, &n, tau, temp, &ltemp, &info);
			if( info )  mexErrMsgTxt( "Error in dschurcguex/dorgqr.");
			nmax = dasum( &dr, h+k, &n);
			j = k;
			for( i=k+1; i<n; i++) {
				tt = dasum( &dr, h+i, &n);
				if( tt > nmax ) {
					nmax = tt;
					j = i;
				}
			}
			if( j > k ) {
				scal = s[k];  s[k] = s[j];  s[j] = scal;
				dswap( &dr, h+k, &n, h+j, &n);
				i = q[k];  q[k] = q[j];  q[j] = i;
			}
		}
		/* l = (G*H(:,k)) ./ ([s(1:k-1);t(k:n)]-s(k)); */
                dgemv( notr, &n, &dr, &one, g, &n, h+k, &n, &zero, l, &ione);
		len = k;
		dcopy( &len, s, &ione, temp, &ione);
		len = n-k;
		dcopy( &len, t+k, &ione, temp+k, &ione);
		daxpy( &n, &minusone, s+k, &izero, temp, &ione);
		dtbsv( lotr, notr, nut, &n, &izero, temp, &ione, l, &ione);
		/* [lmax imax] = max( abs(real(l(k:n)))+abs(imag(l(k:n))) ); */
		len = n-k;
		imax = idamax( &len, l+k, &ione) - 1;
		/* if lmax == 0
			warning('drsolve:clsolve', ...
			'Matrix is singular to working precision.')
		end */
		if( (creal(l[k+imax]) == 0.) && (cimag(l[k+imax]) == 0.) )
			mexWarnMsgIdAndTxt( "drsolve:clsolve:singularMatrix",
				"Matrix is singular to working precision.");
		/* if imax > 1
			imax  = imax+(k-1);
			l([k imax])   = l([imax k]);
			t([k imax])   = t([imax k]);
			G([k imax],:) = G([imax k],:);
			B([k imax],:) = B([imax k],:);
			p([k imax])   = p([imax k]);
		   end
		   pivot = l(k); */
		if( imax > 0 ) {
			imax += k;
			pivot = l[imax];  l[imax] = l[k];
			scal = t[k];  t[k] = t[imax];  t[imax] = scal;
			dswap( &dr, g+k, &n, g+imax, &n);
			dswap( &kb, b+k, &n, b+imax, &n);
			i = p[k];  p[k] = p[imax];  p[imax] = i;
		} else
			pivot = l[k];
		/* l(k) = -1; */
		l[k] = minusone;
		/* u1(1:n-k) = (G(k,:)*H(:,k+1:n)) ./ (t(k)-s(k+1:n).'); */
		len = n-k-1;
		dgemv( notr, &len, &dr, &one, h+k+1, &n, g+k, &n,
			&zero, u1, &ione);
		dcopy( &len, t+k, &izero, temp, &ione);
		daxpy( &len, &minusone, s+k+1, &ione, temp, &ione);
		dtbsv( lotr, notr, nut, &len, &izero, temp, &ione, u1, &ione);
		/* g1_pivot = G(k,:)/pivot; */
		dcopy( &dr, g+k, &n, temp, &ione);
		/* G(k,:) = 0; */
		dcopy( &dr, &zero, &izero, g+k, &n);
		/* G = G-l*g1_pivot; */
		scal = minusone / pivot;
		dger( &n, &dr, &scal, l, &ione, temp, &ione, g, &n);
		/* g2_pivot = B(k,:)/pivot; */
		dcopy( &kb, b+k, &n, temp, &ione);
		/* B(k,:) = 0; */
		dcopy( &kb, &zero, &izero, b+k, &n);
		/* B = B-l*g2_pivot; */
		dger( &n, &kb, &scal, l, &ione, temp, &ione, b, &n);
		/* h1_pivot = H(:,k)/pivot; */
		dcopy( &dr, h+k, &n, temp, &ione);
		/* H(:,k+1:n) = H(:,k+1:n) - h1_pivot*u1(1:n-k); */
		len = n-k-1;
		dger( &len, &dr, &scal, u1, &ione, temp, &ione, h+k+1, &n);
		/* uinvnm = max([uinvnm; ...
		   sum(abs(real(l(1:k)))+abs(imag(l(1:k)))) / abs(pivot) ]); */
		len = k+1;
		uinvnm = max( uinvnm, dasum( &len, l, &ione)/fabs(pivot));
		/* ucolsum(k:n) = ucolsum(k:n) + ...
			[ abs(real(pivot))+abs(imag(pivot)), ...
			abs(real(u1(1:n-k)))+abs(imag(u1(1:n-k))) ]; */
		ucolsum[k] += fabs(creal(pivot))+fabs(cimag(pivot));
		for( i=0; i<n-k-1; i++)
			ucolsum[k+i+1] += fabs(creal(u1[i]))+fabs(cimag(u1[i]));
	}

	/* unorm = max(ucolsum); */
	k = idamax( &n, ucolsum, &ione) - 1;
	/* rcond = 1/unorm/uinvnm; */
	*rcond = one/ucolsum[k]/uinvnm;

	/* B(q,:) = B; */
	for( j=0; j<kb; j++) {
		for( i=0; i<n; i++)
			temp[q[i]-1] = b[i+n*j];
		dcopy( &n, temp, &ione, b+n*j, &ione);
	}

	mxFree(temp);
	mxFree(tau);
	mxFree(ucolsum);
	mxFree(u1);
	mxFree(l);
}

