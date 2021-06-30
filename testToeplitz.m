tol = 1e-12;
hssoption('threshold', tol);
hssoption('block-size', 256);
fun = @expm;

cd util
mex clsolve.c fort.c -lmwblas -lmwlapack
cd ..

ns = 2.^(9:15); % 9-15
maxfull = 2^13; % 13
l = length(ns);

errExpmHss = zeros(l, 1);
timeExpmHss = zeros(l, 1);
timeExpmMatlab = zeros(l, 1);
errDC = zeros(l, 1);
timeDC = zeros(l, 1);
hssRanks = zeros(l, 1);
timeSexpmt = zeros(l, 1);
errSexpmt = zeros(l, 1);

p = [inf]; 


for j = 1:l
    n = ns(j);
    disp(n)
    [C, R] = exampleOptionPricing2(n);
    
    % D&C
    tic;
    H = hss('toeplitz', C, R);
    nrm = ceil(log2(normest(H)));
    F1 = hss_fun_dac(H/2^nrm, fun, p, 0, 1, 0, 0);

    for jjj = 1:nrm
    	F1 = F1 * F1;
    end

    timeDC(j) = toc;
    hssRanks(j) = hssrank(F1);
    
    % expm HSS
    tic;
    H = hss('toeplitz', C, R);
    F2 = expm(H);
    timeExpmHss(j) = toc;
    
    % expmt / sexpmt from [Kressner/Luce] paper
    [G, B, timeSexpmt(j)] = danielrobertexample(n);
    
    % full expm
    if (n <= maxfull)
        T = toeplitz(C, R);
        % disp(bandwidth(double(abs(T) > 1e-6)))
        tic;
        F3 = expm(T);
        timeExpmMatlab(j) = toc;
        

        Fsexpmt = stein_reconstruction(G, B);
        errDC(j) = norm(F3 - full(F1), 'fro')/norm(F3, 'fro');
        errExpmHss(j) = norm(F3 - full(F2), 'fro')/norm(F3, 'fro');
        errSexpmt(j) = norm(F3 - Fsexpmt, 'fro')/norm(F3, 'fro');
    end
end

disp([timeDC, timeExpmHss, timeExpmMatlab, timeSexpmt])
disp([errDC, errExpmHss, errSexpmt])

% dlmwrite('../data/testToeplitz.dat', [ns', timeExpmHss, errExpmHss, timeDC, errDC, timeSexpmt, ...
%     errSexpmt, hssRanks, timeExpmMatlab], '\t');

cd util
delete clsolve.mexa64
cd ..


    
    

