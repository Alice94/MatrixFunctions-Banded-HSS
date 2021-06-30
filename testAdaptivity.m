% General settings
clear all;
close all;
debug = 0;

fontsize = 12;
fontSize = fontsize;
rng(1)

% Sqrt of tridiagonal matrix
n = 2048;
tol = 1e-8;
b = 1;
initial_block_size = max(8, 4*b);
d = linspace(2, 3, n)';
A = spdiags([-ones(n, 1), d, -ones(n, 1)], -1:1, n, n);
f = @(X) sqrtm(full(X));
fvec = @(x) sqrt(full(x));

fprintf('Computing sqrtm(A) for tridiagonal matrix with -1 on sub/super-diagonals and linspace(2, 3, n) on the diagonal\n')
tic;
F = Splitting(A, b, f, initial_block_size, tol, debug);
toc
fA = f(A);
% err = max(max(abs(F - fA)));
err = norm(F - fA, 'fro') / norm(fA, 'fro');
fprintf('Required error was %d, obtained error is %d\n\n', tol, err)

figure('Position', [0 0 500 400])
colormap gray
imagesc(-(F ~= 0))
axis('square')
set(gca,'Fontsize',fontSize);
% saveas(gcf,'adaptiveSpy1','epsc')

figure('Position', [0 0 500 400])
X = 1:n;
% surf(X, X, log(max(abs(fA(X,X)), 10^(-16))/log(10)))
colormap jet
imagesc(log(max(abs(fA), 10^(-16)))/log(10)); colorbar
axis('square')
set(gca,'Fontsize',fontSize);
% saveas(gcf,'adaptiveSpy2','epsc')

%% Comparison with Chebyshev approximation
figure('Position', [0 0 1200 350])
A = spdiags(ones(n, 1) * [1, -2, 1], -1:1, n, n);
f = @expm;
fA = f(full(A));
e = eig(full(A));
sigmamin = min(e);
sigmamax = max(e);

ms = 4:2:40;

% Using FBandedPoly
% Using explicit Chebyshev interpolation polynomials
errFBandedPoly1 = [];
errCheby1 = [];
timeFBandedPoly1 = [];
timeCheby1 = [];
nnzFBandedPoly = [];
nnzCheby = [];

for initial_block_size = b*ms
    tic;
    F1 = Splitting(A, b, f, initial_block_size, 1e30, 0);
    timeFBandedPoly1 = [timeFBandedPoly1, toc];
    errFBandedPoly1 = [errFBandedPoly1, norm(F1 - fA, 'fro')/norm(fA, 'fro')];
    nnzFBandedPoly = [nnzFBandedPoly, nnz(F1)];
    
    tic;
    sigma = 2/(sigmamax - sigmamin);
    tau = (sigmamax + sigmamin) / 2;
    f2 = @(x) f(tau + x / sigma);
    f3 = chebfun(f2, [-1, 1], initial_block_size/b);
    c = chebcoeffs(f3);
    prec2 = speye(n);
    prec1 = sigma * (A - tau * speye(n));
    F2 = c(1) * prec2 + c(2) * prec1;
    for i = 3:length(c)
        current = 2 * sigma * (A - tau * speye(n)) * prec1 - prec2;
        F2 = F2 + c(i) * current;
        prec2 = prec1;
        prec1 = current;
    end
    timeCheby1 = [timeCheby1, toc];
    errCheby1 = [errCheby1, norm(F2 - fA, 'fro')/norm(fA, 'fro')];
    nnzCheby = [nnzCheby, nnz(F2)];
    
end

subplot(1, 3, 1)
semilogy(nnzFBandedPoly, errFBandedPoly1, 'linewidth', 2)
hold on
semilogy(nnzCheby, errCheby1, 'linewidth', 2)
set(gca,'Fontsize',fontsize);
legend('error Algorithm 3', 'error Chebyshev interpolation', 'Location', 'best')
% title('Exp of tridiagonal matrix')
xlabel('nnz')
ylabel('relative error')
hold off

A = spdiags(ones(n, 1) * [1, -2, 1], -1:1, n, n);
A(1, 1) = 10;
f = @expm;
fA = f(full(A));
e = eig(full(A));
sigmamin = min(e);
sigmamax = max(e);

ms = 4:2:40;

% Using FBandedPoly
% Using explicit Chebyshev interpolation polynomials
errFBandedPoly2 = [];
errCheby2 = [];
timeFBandedPoly2 = [];
timeCheby2 = [];
nnzFBandedPoly = [];
nnzCheby = [];

for initial_block_size = b*ms
    tic;
    F1 = Splitting(A, b, f, initial_block_size, 1e30, 0);
    timeFBandedPoly2 = [timeFBandedPoly2, toc];
    errFBandedPoly2 = [errFBandedPoly2, norm(F1 - fA, 'fro')/norm(fA, 'fro')];
    nnzFBandedPoly = [nnzFBandedPoly, nnz(F1)];
    
    tic;
    sigma = 2/(sigmamax - sigmamin);
    tau = (sigmamax + sigmamin) / 2;
    f2 = @(x) f(tau + x / sigma);
    f3 = chebfun(f2, [-1, 1], initial_block_size/b);
    c = chebcoeffs(f3);
    prec2 = speye(n);
    prec1 = sigma * (A - tau * speye(n));
    F2 = c(1) * prec2 + c(2) * prec1;
    for i = 3:length(c)
        current = 2 * sigma * (A - tau * speye(n)) * prec1 - prec2;
        F2 = F2 + c(i) * current;
        prec2 = prec1;
        prec1 = current;
    end
    timeCheby2 = [timeCheby2, toc];
    errCheby2 = [errCheby2, norm(F2 - fA, 'fro')/norm(fA, 'fro')];
    nnzCheby = [nnzCheby, nnz(F2)];
    
end

subplot(1, 3, 2)
semilogy(nnzFBandedPoly, errFBandedPoly2, 'linewidth', 2)
hold on
semilogy(nnzCheby, errCheby2, 'linewidth', 2)
set(gca,'Fontsize',fontsize);
legend('error Algorithm 3', 'error Chebyshev interpolation', 'Location', 'best')
% title('Exp of modified matrix')
xlabel('nnz')
ylabel('relative error')
hold off

d = linspace(2, 3, n)';
A = spdiags([-ones(n, 1), d, -ones(n, 1)], -1:1, n, n);
f = @(X) sqrtm(full(X));
fvec = @(x) sqrt(full(x));
fA = f(full(A));
e = eig(full(A));
sigmamin = min(e);
sigmamax = max(e);

ms = 8:4:80;

% Using FBandedPoly
% Using explicit Chebyshev interpolation polynomials
errFBandedPoly3 = [];
errCheby3 = [];
timeFBandedPoly3 = [];
timeCheby3 = [];
nnzFBandedPoly = [];
nnzCheby = [];

for initial_block_size = b*ms
    tic;
    F1 = Splitting(A, b, f, initial_block_size, 1e30, 0);
    timeFBandedPoly3 = [timeFBandedPoly3, toc];
    errFBandedPoly3 = [errFBandedPoly3, norm(F1 - fA, 'fro')/norm(fA, 'fro')];
    nnzFBandedPoly = [nnzFBandedPoly, nnz(F1)];
    
    tic;
    sigma = 2/(sigmamax - sigmamin);
    tau = (sigmamax + sigmamin) / 2;
    f2 = @(x) f(tau + x / sigma);
    f3 = chebfun(f2, [-1, 1], initial_block_size/b);
    c = chebcoeffs(f3);
    prec2 = speye(n);
    prec1 = sigma * (A - tau * speye(n));
    F2 = c(1) * prec2 + c(2) * prec1;
    for i = 3:length(c)
        current = 2 * sigma * (A - tau * speye(n)) * prec1 - prec2;
        F2 = F2 + c(i) * current;
        prec2 = prec1;
        prec1 = current;
    end
    timeCheby3 = [timeCheby3, toc];
    errCheby3 = [errCheby3, norm(F2 - fA, 'fro')/norm(fA, 'fro')];
    nnzCheby = [nnzCheby, nnz(F2)];
    
end

subplot(1, 3, 3)
semilogy(nnzFBandedPoly, errFBandedPoly3, 'linewidth', 2)
hold on
semilogy(nnzCheby, errCheby3, 'linewidth', 2)
set(gca,'Fontsize',fontsize);
legend('error Algorithm 3', 'error Chebyshev interpolation', 'Location', 'best')
% title('Sqrt of tridiag. matrix with increasing diag')
xlabel('nnz')
ylabel('relative error')
hold off
% saveas(gcf,'cheby2','epsc')

