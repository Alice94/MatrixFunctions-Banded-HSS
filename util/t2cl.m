function [G,H,a,b,xi,eta] = t2cl(c,r,phi,psi)
%T2CL	Converts a Toeplitz matrix to Cauchy-like.
%   [G,H,a,b,xi,eta] = T2CL(c,r) returns the quantities used in the
%   displacement equation
%
%       diag(a) * C - C * diag(b) = G * H'
%
%   of a Cauchy-like matrix C corresponding to the Toeplitz matrix
%   T=TOEPLITZ(c,r).
%   The matrices C and T are such that C = F(xi)'*T*F(eta), where
%   F(xi)=FTIMES(m,[],xi) and F(eta)=FTIMES(n,[],eta) are unitary
%   matrices, and [m,n]=size(T).
%
%   [...] = T2CL(c,r,xi,eta) forces a choice for xi, eta. The default
%   is xi=1 and eta=exp(1i*pi*gcd(m,n)/m). The scalars xi and eta must
%   be unitary, i.e., abs(xi)=abs(eta)=1.
%
%   See also t2tl, tl2cl, ftimes, toeplitz.

%   Antonio Arico' & Giuseppe Rodriguez, University of Cagliari, Italy
%   Email: {arico,rodriguez}@unica.it
%
%   Last revised Mar 25, 2010

% T is m-x-n
% build t = [ T_{1,n} ... T_{1,1} ... T_{m,1} ].' ...
if nargin<2, error('drsolve:t2cl:nargin','too few arguments'), end
c = c(:);
r = r(:);
m = size(c,1);
n = size(r,1);
if r(1) ~= c(1)
    warning('drsolve:t2cl:DiagonalConflict',['First element of ' ...
        'input column does not match first element of input row. ' ...
        '\n         Column wins diagonal conflict.'])
end
t = [r(n:-1:2) ; c];                 % build vector "t" of user data

% define xi eta
if nargin==2
    xi = 1;
    if mod(n,m), eta = exp(complex(0,pi*(gcd(m,n)/m))); else eta = -1; end
    %
elseif nargin~=4 || abs(abs(phi)-1)>eps || abs(abs(psi)-1)>eps
    error('drsolve:t2cl:nargin','check input arguments')
else
    xi=phi; eta=psi; % set up for LHS
end
%if nargout<6 && nargin<4, warning('drsolve:t2cl:output','not injective'), end

% define a b G H
% use +n wrt formula :)
a = nroots1(m,xi);
g = [ ... 
               -eta*t(n)
       t(1:m-1)-eta*t(n+1:n+m-1)
    ];
G = [ ftimes(g,'A',xi), ones(m,1)/sqrt(m) ];
%
b = nroots1(n,eta);
h = [ ...
       conj(xi*t(n+m-1:-1:m+1)-t(n-1:-1:1));
       conj(xi*t(m))
    ];
H = [ exp(complex(0,(angle(eta)*(n-1)-2*pi*(0:n-1))/n)).'/sqrt(n), ftimes(h,'A',eta) ];
