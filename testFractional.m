hssoption('threshold', 1e-8);
hssoption('block-size', 256);
do_print = 1;
sizes = 2.^[9:15]; %sizes = 2^12;
l = length(sizes); %l = 1;
errors = zeros(l, 2); % first column error of expm, second column error of dac
times = zeros(l, 2);  % first column time of expm, second column time of dac
ranks = zeros(l, 2);  % first column hssrank of A, second column hssrank of f(A)
times_dense = zeros(l, 1);
alpha = 1.2;
poles = inf; 
for j = 1:l
	n = sizes(j);
	h = 2 / (n + 2); % space discretization step
    ni = h^(alpha - 1); % time step is assumed dt equal to h
    t = linspace(h, 2 - h, n);
    d1 = ones(1, n);
    d2 = ones(1, n);
    [am, ap] = fractional_symbol(alpha, n);

    hssA = -hss('toeplitz', am, ap, n) / h^(alpha - 1);
    hssA = hss('diagonal', d1) * hssA + hss('diagonal', d2) * hssA';
	hssA = (hssA + hssA') / 2;
	nrmfA = exp(eigs(hssA, 'opts', 'sm', 'k', 1));

	ranks(j, 1) = hssrank(hssA);
	tic
	sA = expm(hssA);
	times(j, 1) = toc;
	tic
%profile on
	sA2 = hss_fun_dac(hssA, @expm, poles, 0, nrmfA, 1);
%profile viewer
%return
	times(j, 2) = toc;
	ranks(j, 2) = hssrank(sA2);
	
	if n <= 8192 
		A = -toeplitz(am, [ap, zeros(1, length(am) - 2)]) / h^(alpha - 1);
		A = diag(d1) * A + diag(d2) * A';
		A = (A + A') / 2;
		tic
		fA = expm(A); 
		times_dense(j) = toc;
		errors(j, 1) = norm(fA - full(sA), 'fro') / norm(fA, 'fro');
		errors(j, 2) = norm(fA - full(sA2), 'fro') / norm(fA, 'fro');
		if do_print
			fprintf('n = %d, hssrank = %d,\t  err expm = %1.2e,  err dac = %1.2e,  Time expm = %.2f,  Time dac = %.2f, Time dense = %.2f, hssrank = %d\n', n, ranks(j, 1), errors(j, 1), errors(j, 2), times(j, 1), times(j, 2), times_dense(j), ranks(j, 2));
		end
	else
		if do_print
			fprintf('n = %d, hssrank = %d,\t err expm = --------,  err dac = --------,  Time expm = %.2f,  Time dac = %.2f, hssrank = %d \n', n, ranks(j, 1), times(j, 1), times(j, 2), ranks(j, 2));
		end
	end
end

%dlmwrite('../data/testfractional.dat', [sizes', times, errors, ranks, times_dense], '\t');

%----------Auxiliary function-------------------------
function [am, ap] = fractional_symbol(alpha, n)
%FRACTIONAL_SYMBOL Construct the symbol of the Grunwald-Letkinov derivative
%
% [AM, AP] = FRACTIONAL_SYMBOL(ALPHA, N) construct the negative and
%     positive parts of the symbol of the Toeplitz matrix discretizing the
%     fractional derivative by means of the Grunwald-Letkinov shifted
%     formulas. 
%
	v = zeros(n+2, 1);
	v(1) = 1;
	v = -cumprod([1,1-((alpha + 1)./(1:n))]);
	am = v(2:end);
	ap = [v(2), v(1)];
end


