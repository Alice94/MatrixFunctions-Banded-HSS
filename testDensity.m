clear all;
close all;
rng(1);
hssoption('threshold', 1e-8);
hssoption('block-size', 256);

D = 1e-1;
delta = 1e-4;
Ns = 5;
Nb = 1600;
C = 1e-1;

mu = 0.0;
poles = [0 inf];
debug = 0;
err_diags = zeros(4, 1);
err_traces = zeros(4, 1);
t_diags = zeros(4, 1);
t_traces = zeros(4, 1);
t_true_diags = zeros(4, 1);
t_true_traces = zeros(4, 1);
fun = @(M) inv_sqrt(M);
fun2 = @(M) my_signm(M);
fun3 = @(M) my_signv(M);
do_print = 1;
it = 1;
for nod = 5000%[5 50 500 5000]
	% Costruction of the Hamiltonian TODO: optimize (for dimension Nb * Ns = 2000 is ok)
	A = zeros(Nb * Ns);
	for i = 1:Nb
		for ii = 1:Nb
			for j = 1:Ns
				for jj = 1:Ns
					if i == ii && j == jj 	% diagonal element
						A((i - 1) * Ns + j, (i - 1) * Ns + j) = (i - 1) * D + (j - 1) * delta;
					elseif i == ii % intraband elements
						A((i - 1) * Ns + j, (i - 1) * Ns + jj) = C * exp(-abs(j - jj));
					elseif i ~= ii % interband element
						A((i - 1) * Ns + j, (ii - 1) * Ns + jj) = C * exp(-abs(j - jj)) / (nod * (abs(i - ii) + 1));
					end 
				end
			end
		end
    end
	
	A = (A + A') / 2;
	l = eig(A);

	hssA = hss(A - mu*eye(Ns*Nb));
	hssA = (hssA + hssA') / 2;


	%fA = signm(A);
	fA = heavys(A, mu);
	normfA = norm(fA, 'fro')^2;
	true_diag = diag(fA);
	true_trace = trace(fA);
    semilogy(true_diag)
    tic
    diag(heavys(A, mu));
    t_true_diags(it) = toc
    tic
    sum(heavysv(eig(A), mu));
    t_true_traces(it) = toc
    tic;
    hss_diag = hss_fun_dac(hssA, fun2, poles, debug, normfA, 1, 1);
    hss_diag = (ones(Nb * Ns, 1) - hss_diag)/2;
    t_diags(it) = toc

    tic;
    hss_trace = hss_fun_dac(hssA, fun3, poles, debug, normfA, 1, 2);
    hss_trace = (Nb * Ns - hss_trace) / 2;
    t_traces(it) = toc


	err_diags(it) = norm(true_diag - hss_diag) / norm(true_diag);
	err_traces(it) = abs(true_trace - hss_trace) / norm(true_trace);

	if do_print
		fprintf('nod = %d, err diag = %1.2e, Time hss_diag = %.4f, Time dense_diag = %.2f, true trace = %.2f, err trace = %1.2e,   Time hss_trace = %.4f, Time dense_trace = %.2f\n', nod, err_diags(it), t_diags(it), t_true_diags(it), true_trace, err_traces(it), t_traces(it), t_true_traces(it));
	end
	it = it + 1;

end

% dlmwrite('testdensity.dat', [[5,50,500,5000]', err_diags, t_diags, t_true_diags, err_traces, t_traces, t_true_traces], '\t');



%----------------- Auxiliary function --------------
function X = signm(A)
	if ~ishermitian(A)
		error('SIGNM:: not Hermitian argument')
	end
	[V, D] = eig(A, 'vector');
	ind1 = find(D > 0);
	ind2 = find(D < 0);
	X = V(:, ind1) * V(:, ind1)' - V(:, ind2) * V(:, ind2)';
end
function F = heavys(A, mu)
	[V, D] = eig(A, 'vector');
	ind = find(D < mu);
	%F = V * diag(D) * V';
	F = V(:, ind) * V(:, ind)';
end

function v = heavysv(y, mu)
	v = (y < mu);
end
function F = inv_sqrt(A)
	[V, D] = eig(A, 'vector');
	D = 1./sqrt(D);
	F = V * diag(D) * V';
end

function F = my_signm(A)
	[V, D] = eig(A, 'vector');
	D = D./sqrt(D.^2);
	F = V * diag(D) * V';
end
function F = my_signv(D)
	F = D./sqrt(D.^2);
end
