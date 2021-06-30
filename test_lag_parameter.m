% This needs METIS + metismex, see https://github.com/dgleich/metismex

tol = 1e-8;

% Graphs
lags = 1:4;
warning('off')
file = "nopoly.mat";
load(file);
A = Problem.A;
n = size(A, 1);
lambdamax = max(eigs(A));

timeDCDiag1 = [];
timeDCTrace1 = [];
errDCDiag1  = [];
errDCTrace1 = [];
debug = 0;
minblocksize = 256;

d2 = diag(expm(A));

for lag = lags
    tic;
    d1 = compute_diag(A, exp(lambdamax), debug, minblocksize, 0, tol, 1, lag);
    timeDCDiag1 = [timeDCDiag1, toc];
    errDCDiag1 = [errDCDiag1, norm(d1-d2)/norm(d2)];
end
disp(timeDCDiag1)
disp(errDCDiag1)

for lag = lags
    tic;
    t1 = compute_diag(A, exp(lambdamax), debug, minblocksize, 1, tol, 1, lag);
    timeDCTrace1 = [timeDCTrace1, toc];
    errDCTrace1 = [errDCTrace1, abs(t1 - sum(d2))/sum(d2)];
end
disp(timeDCTrace1)
disp(errDCTrace1)

file = "worms20_10NN.mat";
load(file);
A = Problem.A;
n = size(A, 1);
lambdamax = max(eigs(A));

timeDCDiag2 = [];
timeDCTrace2 = [];
errDCDiag2  = [];
errDCTrace2 = [];
debug = 0;
minblocksize = 500;

d2 = diag(expm(A));

for lag = lags
    tic;
    d1 = compute_diag(A, exp(lambdamax), debug, minblocksize, 0, tol, 1, lag);
    timeDCDiag2 = [timeDCDiag2, toc];
    errDCDiag2 = [errDCDiag2, norm(d1-d2)/norm(d2)];
end
disp(timeDCDiag2)
disp(errDCDiag2)

for lag = lags
    tic;
    t1 = compute_diag(A, exp(lambdamax), debug, minblocksize, 1, tol, 1, lag);
    timeDCTrace2 = [timeDCTrace2, toc];
    errDCTrace2 = [errDCTrace2, abs(t1 - sum(d2))/sum(d2)];
end
disp(timeDCTrace2)
disp(errDCTrace2)

% file = "fe_body.mat";
% load(file);
% A = Problem.A;
% lambdamax = max(eigs(A));
% 
% timeDCDiag3 = [];
% timeDCTrace3 = [];
% debug = 0;
% minblocksize = 500;
% 
% for lag = lags
%     tic;
%     d1 = compute_diag(A, exp(lambdamax), debug, minblocksize, 0, tol, 1, lag);
%     timeDCDiag3 = [timeDCDiag3, toc];
% end
% disp(timeDCDiag3)
% 
% for lag = lags
%     tic;
%     t1 = compute_diag(A, exp(lambdamax), debug, minblocksize, 1, tol, 1, lag);
%     timeDCTrace3 = [timeDCTrace3, toc];
% end
% disp(timeDCTrace3)

dlmwrite('../data/testLag.dat', [lags', timeDCDiag1', timeDCTrace1', errDCDiag1', errDCTrace1', ...
    timeDCDiag2', timeDCTrace2', errDCDiag2', errDCTrace2'], '\t');

warning('on')


% Useful stuff

function Y = invsqrt(X)
    [V, D] = eig(full(X));
    d = diag(D);
    d = -1i*d;
    d = sqrt(d);
    d = (1/sqrt(2) + 1i/sqrt(2))*d;
    d = 1./d;
    Y = V * diag(d) / V;
end

% %% Neumann-to-Dirichlet problem
% n = 2^14;
% k = 50;
% tol = 1e-8;
% hssoption('threshold', tol);
% hssoption('block-size', 512);
% f = @invsqrt;
% 
% lags = 1:3;
% l = length(lags);
% timeDC = [];
% 
% ns = 2^14;
% debug = 0;
% 
% h = pi/(n+1);
% A = 1/h^2 * spdiags(ones(n, 1) * [-1, 2, -1], -1:1, n, n) - k^2 * speye(n);
% 
% b = normest(A, 1e-2);
% a = b / condest(A);
% 
% m = 6;
% s = zeros(1, m);
% deltaprime = sqrt(1 - (a/b));
% for i = 1:m
%     [sn, ~, dn] = ellipj((2*m-2*i+1)*ellipke(deltaprime^2)/(2*m), deltaprime^2);
%     s(i) = sqrt(b) * dn;
% end
% 
% h = poly([s, 1i*s]) * (-1i)^m;
% pn = -h(2:2:end);
% qn = h(1:2:end);
% 
% p = roots(qn)';
% 
% for lag = 1 %lags
%     disp(lag)
%     profile on
%     tic;
%     Y = hss_fun_dac_invsqrt(A, f, p, debug, 1, lag);
%     timeDC = [timeDC, toc]
%     profile viewer
% end
