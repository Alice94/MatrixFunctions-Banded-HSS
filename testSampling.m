rng(1);
hssoption('threshold', 1e-8);
hssoption('block-size', 256);
do_print = 1;
debug = 0;

sizes = 2.^[9:18]; 
l = length(sizes); 
deltas = 2e-2 * 2.^[0 : -1 : 1 - l];

phi = 3;
fun = @(M) inv_sqrt(M);
errors = zeros(l, 2); % first column error of sqrtm, second column error of dac
times = zeros(l, 2);  % first column time of sqrtm, second column time of dac
ranks = zeros(l, 2);  % first column hssrank of A, second column hssrank of f(A)
times_dense = zeros(l, 1);
band = zeros(l, 1);   % bandwidth of A
poles = [0 inf];% poles for the Krylov projection method
for j = 1:l
	n = sizes(j);
	delta = deltas(j);
	s = sort(rand(n, 1));

	% Construct the precision matrix A
	A = sparse(n, n);
	for b = 1:n
		x = abs(s(1+b:end) - s(1:end-b)) < delta;
		if  sum(x) == 0
			break
		else
			A = A + spdiags([x, zeros(n - b, 1)], -b, n, n);
			b = b + 1;
		end
	end
	A = A + A';
	D = phi * A * ones(n, 1) + ones(n, 1);
	A = spdiags(D, 0, n, n) - phi * A;
	nrmfA = 1.0; %inv_sqrt(eigs(A, 1, 'smallestabs','MaxIterations', 1e5))
	
	band(j) = bandwidth(A);
	hssA = hss(A); 
	hssA = (hssA + hssA')/2;

	ranks(j, 1) = hssrank(hssA);
	tic
	[~, sA] = sqrtm(hssA);
	times(j, 1) = toc;

	tic
    % profile on
	sA2 = hss_fun_dac_band_hermitian(A, fun, poles, debug, nrmfA, 0);
    % profile off
	times(j, 2) = toc;
	ranks(j, 2) = hssrank(sA2);
	
	if n <= 8192 
		fA = full(A); fA = (fA + fA')/2;
		tic
		fA = inv_sqrt(fA); 
		times_dense(j) = toc;
		errors(j, 1) = norm(fA - full(sA), 'fro') / norm(fA, 'fro');
		errors(j, 2) = norm(fA - full(sA2), 'fro') / norm(fA, 'fro');
		if do_print
			fprintf('n = %d, band = %d, hssrank = %d,\t  err sqrtm = %1.2e,  err dac = %1.2e,  Time sqrtm = %.2f,  Time dac = %.2f, Time dense = %.2f, hssrank = %d\n', n, band(j), ranks(j, 1), errors(j, 1), errors(j, 2), times(j, 1), times(j, 2), times_dense(j), ranks(j, 2));
		end
	else
		if do_print
			fprintf('n = %d, band = %d, hssrank = %d,\t err sqrtm = --------,  err dac = --------,  Time sqrtm = %.2f,  Time dac = %.2f,  hssrank = %d \n', n, band(j), ranks(j, 1), times(j, 1), times(j, 2), ranks(j, 2));
		end
	end
% pause
end

% dlmwrite('../data/testsampling.dat', [sizes', times, errors, band, ranks, times_dense], '\t');

%----------------- Auxiliary function --------------
function F = inv_sqrt(A)
    A = (A + A')/2;
	[V, D] = eig(A, 'vector');
	D = 1./sqrt(D);
	F = V * diag(D) * V';
end

