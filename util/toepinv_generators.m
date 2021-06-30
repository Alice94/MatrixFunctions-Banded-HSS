function [Ginv, Binv] = toepinv_generators(c, r)
% [Ginv, Binv] = toepinv_generators(c, r)
%
% Computes a generator for the inverse of a Toeplitz matrix, with respect
% to the Stein displacement equation T - Z*T*Z', with Z being the downshift
% matrix.

n = length(c);
assert(length(r) == n && c(1) == r(1));

% Match notation in the paper
g = c(:);
b = conj(r(:));
b(1) = 0.;

e1 = zeros(n,1);
e1(1) = 1.0;

Ginv = solve_one(c,r, [-g, e1], false);
Ginv(1,1) = Ginv(1,1) + 1;

Binv = solve_one(c,r, [e1, -b], true);
Binv(1,2) = Binv(1,2) + 1;

end

function x = solve_one(c, r, rhs, trans)
% Compute (Z-I) * T(c,r)^{-1} * (I-Z)^{-1} * rhs
% or the same system with T^* if 'trans' is true.

x = -vapply(rhs, 'inv');
if ~trans
    x = -toepsolve(c,r,x);
else
    x = -toepsolve(conj(r),conj(c),x);
end
x = vapply(x);

end


