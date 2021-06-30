function x = toepsolve(c, r, b)
% x = toepsolve(c, r, b)
%
% Solve a Toeplitz system of equations in O(n^2) using drsolve package.

n = length(c);

if n==1
    if r ~= c
        warning('Inconsistent Toeplitz data, column information wins');
    end
    % Special case not treated by drsolve's tsolve
    if c==0.0
        x = NaN;
    else
        x = b/c;
    end
    return;
end

% Pass through to drsolve Toeplitz solver.
x = tsolve(c,r,b,1);
%return;

% This does one step of iterative refinement, could be made the default
for ii=1:1
res = toepmult(c,r,x) - b;
xx = tsolve(c,r,res,1);
x = x - xx;
end
end