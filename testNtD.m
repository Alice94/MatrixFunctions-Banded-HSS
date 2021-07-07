addpath('util')
addpath('util/matrices')

ns = 2.^(9:15); % 9-15
maxfull = 2^13; % 13
k = 50;
tol = 1e-8;
hssoption('threshold', tol);
hssoption('block-size', 256);
f = @invsqrt;

l = length(ns);
errDC = zeros(l, 1);
timeDC = zeros(l, 1);
hssRanks = zeros(l, 1);
timeFull = zeros(l, 1);

debug = 0;
lag = 1;


for j = 1:l
    n = ns(j);
    disp(n)
    h = pi/(n+1);
    A = 1/h^2 * spdiags(ones(n, 1) * [-1, 2, -1], -1:1, n, n) - k^2 * speye(n);

    tic;
    Y = hss_fun_dac_invsqrt(A, f, debug, 1, lag);
    timeDC(j) = toc;
    hssRanks(j) = hssrank(Y);
    fprintf('Time %d for divide & conquer, hss rank %d\n', timeDC(j), hssRanks(j))
    
    if (n <= maxfull)
        tic;
        fA = f(A);
        timeFull(j) = toc;
        
        errDC(j) = norm(full(Y) - fA, 'fro') / norm(fA, 'fro');
        fprintf("Error Divide & Conquer = %1.2e \n", errDC(j));
    end
end

% dlmwrite('../data/testNtD.dat', [ns', timeDC, errDC, timeFull, hssRanks], '\t');


function Y = invsqrt(X)
    [V, D] = eig(full(X));
    d = diag(D);
    d = -1i*d;
    d = sqrt(d);
    d = (1/sqrt(2) + 1i/sqrt(2))*d;
    d = 1./d;
    Y = V * diag(d) / V;
end