debug = 0;
tol = 1e-8;
b = 1;
initial_block_size = max(64, 4*b);
hssoption('threshold', tol);
hssoption('block-size', 128);
nsfA = 2.^(8:13);
nsF = [nsfA, 2.^(14:18)];
f = @expm;
fvec = @exp;

timeFBandedPoly = [];
timefA = [];
timeHSS = [];

for n = nsF
    A = spdiags(ones(n, 1) * [1, -2, 1], -1:1, n, n);
    if (n < 4000)
        timeFBandedPoly = [timeFBandedPoly, timeit(@() Splitting(A, b, f, initial_block_size, tol, 0) )];
    
    else
        tic;
        F = Splitting(A, b, f, initial_block_size, tol, 0);
        timeFBandedPoly = [timeFBandedPoly, toc];
    end
    fprintf("FBandedPoly with size %d took %f seconds\n", n, timeFBandedPoly(end));
%     fprintf("Error = %d\n", norm(full(F) - expm(full(A)), 'fro'));
end

for n = nsF
    A = spdiags(ones(n, 1) * [1, -2, 1], -1:1, n, n);
    g = @(X) expm(-X);
    if (n < 4000)
        timeHSS = [timeHSS, timeit(@() hss_fun_dac_band_hermitian(-A, g, inf, debug, 1, 0))];
    else
    tic;
        X = hss_fun_dac_band_hermitian(-A, g, inf, debug, 1, 0);
        timeHSS = [timeHSS, toc];
    end
    fprintf("HSS with rank-b updates with size %d took %f seconds\n", n, timeHSS(end));
    if (n < 2000)
        disp(norm(expm(A) - full(hss_fun_dac_band_hermitian(-A, g, inf, debug, 1, 0)), 'fro'))
    end
end


for n = nsfA   
    A = spdiags(ones(n, 1) * [1, -2, 1], -1:1, n, n);
    if (n < 1025)
        timefA = [timefA, timeit(@() f(full(A)))];
    else
        tic;
        fA = f(full(A));
        timefA = [timefA, toc];
    end
    fprintf("expm(full(A)) with size %d took %f seconds\n", n, timefA(end));
end

loglog(nsfA, timefA, '-d', 'Linewidth', 2)
hold on
loglog(nsF, timeFBandedPoly, '-*', 'Linewidth', 2);
loglog(nsF, timeHSS, '-o', 'Linewidth', 2);
loglog(nsF, nsF * 2 *timeFBandedPoly(end) / nsF(end), ':', 'Linewidth', 2);
loglog(nsF, 2 * nsF .*  log(nsF) * timeHSS(end) / (nsF(end) * log(nsF(end))),':', 'Linewidth', 2);
xlabel('n')
ylabel('time')
legend('Matlab expm', 'Algorithm 4', 'Algorithm 2 (banded case)', 'O(n) bound', 'O(n log(n)) bound', 'Location', 'best')


% saveas(gcf,'timesPolySymmetric','epsc')
hold off