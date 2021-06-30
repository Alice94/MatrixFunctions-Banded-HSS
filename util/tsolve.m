function [x rcond] = tsolve(c,r,b,piv)
%TSOLVE	Solution of a Toeplitz linear system.
%   x = TSOLVE(c,r,b) computes the solution to the linear system T*x=b,
%   where T = TOEPLITZ(c,r) is a square nonsingular matrix. b may
%   contain multiple columns.
%   TSOLVE converts the Toeplitz system to a Cauchy-like system, and
%   then solves it by CLSOLVE.
%
%   x = TSOLVE(c,r,b,piv) calls CLSOLVE with the parameter piv, which
%   selects a pivoting technique. The default is piv=1 (partial
%   pivoting). See CLSOLVE for further details on pivoting.
%
%   [x,rcond] = TSOLVE(...) also returns the estimate for the reciprocal
%   of the condition number given by CLSOLVE.
%
%   See also clsolve, tlsolve.

%   Antonio Arico' & Giuseppe Rodriguez, University of Cagliari, Italy
%   Email: {arico,rodriguez}@unica.it
%
%   Last revised Mar 25, 2010

if nargin<3, error('drsolve:tsolve:nargin','too few arguments'), end
if nargin<4, piv = 1; end
c = c(:);
r = r(:);
n = size(c,1);
if (n~=size(r,1)) || (n~=size(b,1)) 
    error('drsolve:tsolve:size','check input arguments [c r b].')
end

reale = isreal(c) && isreal(r) && isreal(b);

[GC,HC,tC,sC,xi,eta] = t2cl(c,r);
% now xi is 1 and eta is -1

bC        = ftimes(b,'A',xi);
[x rcond] = clsolve(GC,HC,tC,sC,bC,piv);
x         = ftimes(x,'N',eta);

if reale, x = real(x); end

%if rcond < eps
%	warning('drsolve:tsolve:badConditioning',...
%    ['Matrix is close to singular or badly scaled.\n' ...
%    '         Results may be inaccurate. RCONDU = %e.'], rcond)
%end

