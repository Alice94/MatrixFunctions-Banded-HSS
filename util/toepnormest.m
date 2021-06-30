function [est, numiter] = toepnormest(c, r)

% est = toepnormest(c,r)
n = length(c);

% This is rather shaky, should be caller settable or so
% Stop if relative change is smaller than this
eps_tol = 1e-2;
maxiter = 25;

% Start in range(T)
x = toepmult(r', c', ones(n,1));
numiter = 0;
est = norm(x);
x = x/est;

est_old = 0;
while abs(est-est_old) > eps_tol * est && numiter < maxiter
    numiter = numiter+1;
    est_old = est;

    Tx = toepmult(c, r, x);
    x = toepmult(r', c', Tx);
    norm_x = norm(x);
    
    % TODO Could take the max of all three intermediate ratios.
    est = norm_x/norm(Tx);
    x = x/norm_x;    
end

if maxiter == numiter
    % Should not happen, otherwise increase iter limit
    assert(false);
end

end