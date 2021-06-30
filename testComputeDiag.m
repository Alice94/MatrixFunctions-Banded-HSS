% This needs METIS + metismex, see https://github.com/dgleich/metismex

files = ["minnesota.mat", "power.mat", "as-735.mat", "nopoly.mat", "worms20_10NN.mat", "fe_body.mat"];

ns = [];

timeDCDiag1 = []; % with mmq on lowest levels
timeDenseDiag = [];
timeMMQDiag = [];
errDCDiag1 = [];
errMMQDiag = [];
diffDCmmq = [];

timeDCTrace = [];
timeDenseTrace = [];
errDCTrace = [];

maxfull = 24000;
minblocksize = 256;
debug = 0;
tol = 1e-8;

for file = files
    disp(file)
    load(file);
    A = Problem.A;
    % A = A/normest(A, 1e-2);
    lambdamax = max(eigs(A))
    n = size(A, 1);
    disp(n)
    ns = [ns, n];
    
    % Computation of the diagonal
    % D&C for the diagonal with mmq on small blocks
    tic;
    d1 = compute_diag(A, exp(lambdamax), debug, minblocksize, 0, tol, 1, 1);
    timeDCDiag1 = [timeDCDiag1, toc]
    
    % mmq
    tic;
    d2 = zeros(n, 1);
    for i = 1:n
        [d2(i), ~] = mmq_exp_given_tol(A,i,tol);
    end
    timeMMQDiag = [timeMMQDiag, toc]
    diffDCmmq = [diffDCmmq, norm(d1-d2)/norm(d2)]
    
    % Dense diag
    if (n <= maxfull)
        tic;
        d3 = diag(expm(A));
        timeDenseDiag = [timeDenseDiag, toc]

        errDCDiag1 = [errDCDiag1, norm(d1-d3)/norm(d3)]
        errMMQDiag = [errMMQDiag, norm(d2-d3)/norm(d3)]
    else
        timeDenseDiag = [timeDenseDiag, -1]
        errDCDiag1 = [errDCDiag1, -1]
        errMMQDiag = [errMMQDiag, -1]
    end
    
%     Computation of the trace
%     Divide and conquer
    tic;
    t0 = compute_diag(A, exp(lambdamax), debug, minblocksize, 1, tol, 0, 1);
    timeDCTrace = [timeDCTrace, toc]
    
    % Dense trace
    if (n <= maxfull)
        tic;
        t1 = sum(exp(eig(full(A))));
        timeDenseTrace = [timeDenseTrace, toc]
        
        errDCTrace = [errDCTrace, abs(t1-t0)/abs(t1)]
    else
        errDCTrace = [errDCTrace, -1]
        timeDenseTrace = [timeDenseTrace, -1]
    end 
end

format shorte
disp([errDCDiag1', errMMQDiag']);
format shorte
disp(errDCTrace')

% dlmwrite('../data/testDiagGraphsNew.dat', [ns',timeDCDiag1', errDCDiag1', ...
%     timeMMQDiag', errMMQDiag', timeDenseDiag', timeDCTrace', errDCTrace', timeDenseTrace'], '\t');
% save("playComputeDiag.mat");

format short
disp([timeDCDiag1', timeMMQDiag', timeDenseDiag', timeDCTrace', timeDenseTrace']);

