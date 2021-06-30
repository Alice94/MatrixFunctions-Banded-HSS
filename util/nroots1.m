function c = nroots1(n,xi)
%NROOTS1 n-th roots of unity.
%   c = NROOTS1(n,xi) computes the n-th roots of xi on the unit circle.
%
%   See also angle.

%   Antonio Arico' & Giuseppe Rodriguez, University of Cagliari, Italy
%   Email: {arico,rodriguez}@unica.it
%
%   Last revised May 18, 2008

if abs(abs(xi)-1) > eps
	error('drsolve:nroots1','Argument is not on the unit circle')
end

c = exp(complex(0,(angle(xi)+2*pi*(0:n-1)')/n));
