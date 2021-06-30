function y = toepmult(c,r,x)
% y = toepmult(c,r,x)
%
% Compute y = T(c,r) * x
%
% NOTE: trivial wrapper around ttimes from drsolve package.

y = ttimes(c,r,x);

end