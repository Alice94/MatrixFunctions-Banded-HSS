addpath('util')
addpath('util/matrices')

rng(1);
debug = 2;
minbs = 32;
tol = 1e-5;
maxfull = 2^14;

sizes = 2.^[9:19]; 
l = length(sizes);
beta = 1.84;
mu = 0.5;

f = @(M) inv(expm(beta * (M - mu * eye(size(M,1)))) + eye(size(M,1)));

errors = zeros(l, 1); % first column error of sqrtm, second column error of dac
times = zeros(l, 1);  % first column time of sqrtm, second column time of dac
band = zeros(l, 1);  % first column hssrank of A, second column hssrank of f(A)
times_dense = zeros(l, 1);
timesCheby = zeros(l, 1);
errorsCheby = zeros(l, 1);

% Prepare stuff for Chebyshev
sigmamin = -2;
sigmamax = 3;
sigma = 2/(sigmamax - sigmamin);
tau = (sigmamax + sigmamin) / 2;
f2 = @(x) f(tau + x / sigma);


for j = 1:l
	n = sizes(j);
	A = spdiags([-ones(n, 1), rand(n, 1), -ones(n, 1)], -1:1, n, n);
    
    % Splitting algorithm
    tic;
    fA1 = Splitting(A,1,f,minbs,tol,debug);
    times(j) = toc;
	band(j) = nnz(fA1)/n;
    
    % Chebyshev interpolation with band b
    tic; 
    f3 = chebfun(f2, [-1, 1], ceil(band(j)/2));
    c = chebcoeffs(f3);
    %tic; % This should include the above part, which is independent on n actually
    prec2 = speye(n);
    prec1 = sigma * (A - tau * speye(n));
    fA2 = c(1) * prec2 + c(2) * prec1;
    for i = 3:length(c)
        current = 2 * sigma * (A - tau * speye(n)) * prec1 - prec2;
        fA2 = fA2 + c(i) * current;
        prec2 = prec1;
        prec1 = current;
    end
    timesCheby(j) = toc;
	
    
    % Full computations
	if n <= maxfull 
		tic
		fA = f(A); 
        fprintf('Norm of f(A) is %.2f\n', norm(fA, 'fro'));
		times_dense(j) = toc;
		errors(j) = norm(fA - fA1, 'fro')/norm(fA, 'fro');
        errorsCheby(j) = norm(fA - fA2, 'fro')/norm(fA, 'fro');
		fprintf('n = %d, \t  error = %1.2e,  errorCheby = %1.2e, Time fast = %.2f, Time Cheby = %.2f, Time dense = %.2f, band = %.2f\n', ...
            n, errors(j), errorsCheby(j), times(j), timesCheby(j), times_dense(j), band(j));
    else
		fprintf('n = %d, \t Time fast = %.2f, Time Cheby = %.2f, band = %.2f\n', n, times(j), timesCheby(j), band(j));
	end
% pause
end

% dlmwrite('../data/testfermidirac.dat', [sizes', times, errors, band, times_dense, timesCheby, errorsCheby], '\t');