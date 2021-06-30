function X = hss_fun_dac_invsqrt(A, f, debug, normfA, lag)
% A is a tridiagonal matrix
% Computes f(A) in the HSS format for an HSS A, or diag(f(A)), or trace(f(A))
%----------------------------INPUT-----------------------------------------------------------------------------------
%
% A			HSS matrix
% f			handle function that computes the function on a matrix argument (e.g. @expm)
% poles		shift parameters for the Krylov method used for computing the update
% debug		allows some debugging print
% normfA	estimate of the norm of fA 
% isherm	flag that indicates whether the argument is hermitian
% onlyd  	flag that denotes whether only the diagonal (onlyd = 1) or the trace (onlyd = 2) is computed. 
%			In the case of the trace, f should apply entrywise to a vector argument
%
%----------------------------OUTPUT----------------------------------------------------------------------------------
%
% X			HSS representation of f(A);
%
%--------------------------------------------------------------------------------------------------------------------
%
% This is for the invsqrt case, in which even when A is Hermitian, f(A) is
% NOT Hermitian in general.
%
%--------------------------------------------------------------------------------------------------------------------
    
    % Construct poles
    b = normest(A, 1e-2);
    a = b / condest(A);
    m = 6;
    s = zeros(1, m);
    deltaprime = sqrt(1 - (a/b));
    for i = 1:m
        [~, ~, dn] = ellipj((2*m-2*i+1)*ellipke(deltaprime^2)/(2*m), deltaprime^2);
        s(i) = sqrt(b) * dn;
    end
    
    h = poly([s, 1i*s]) * (-1i)^m;
    qn = h(1:2:end);
    poles = roots(qn)';

	if ~exist('debug', 'var')
		debug = 0;
	end
	if ~exist('normfA', 'var')
		normfA = 1;
    end

	tol = normfA * hssoption('threshold'); % set relative accuracy

	[m, n] = size(A); 
	if m ~= n
		error('HSS_FUN_DAC:: the input argument is not a square matrix');
	end

	% Base case of the recursion
	if size(A, 1) <= hssoption('block-size')
        X = hss();
        X.D = f(full(A));
        X.topnode = 1;
        X.leafnode = 1;
        return
    end
    
    mid = floor(n/2);
    DA1 = A(1:mid, 1:mid);
    DA2 = A(mid+1:end, mid+1:end);
	DA = blkdiag(DA1, DA2);

	% Recursive call on the diagonal blocks
    X = blkdiag(hss_fun_dac_invsqrt(DA1, f, debug, normfA, lag), ...
        hss_fun_dac_invsqrt(DA2, f, debug, normfA, lag));

%------------------------DEBUG--------------------------------------------------------------------------------------------
	if debug == 1
		fprintf('--------- block size: %d, cond(A) = %1.2e, sought accuracy = %1.2e ---------\n', n, cond(full(A)), tol)
		fA = full(A); 
		fDA1 = full(DA1); fDA2 = full(DA2);
		fDA1 = (fDA1 + fDA1') / 2; 	fDA2 = (fDA2 + fDA2') / 2; fA = (fA + fA') / 2;
		trueDF = f(fA) - blkdiag(f(fDA1), f(fDA2));
	end
	if debug == 2 || debug == 3
		fprintf('--------- block size: %d, sought accuracy = %1.2e ---------\n', n, tol)
	end
%-----------------------END DEBUG----------------------------------------------------------------------------------------

	DFold = {};
			
    % Retrieve the low-rank factorization of the offdiagonal part (W*Z^T)
    W = [1; zeros(n-mid-1, 1)];
    Z = [zeros(mid-1, 1); 1];
    B = [0, A(mid+1, mid); A(mid, mid+1), 0];	
    k = 1;
		
    % Compute the correction with a Krylov method
    maxit = ceil( (n - 1) / (length(poles)));
    for j = 1:maxit
        % Rational Krylov that repeats cyclically the poles provided by the user
        if j == 1
            [VV1, K1, H1, params1] = rat_krylov(DA1, Z, poles);
            [VV2, K2, H2, params2] = rat_krylov(DA2, W, poles);
        else
            [VV1, K1, H1, params1] = rat_krylov(DA1, VV1, K1, H1, poles, params1);
            [VV2, K2, H2, params2] = rat_krylov(DA2, VV2, K2, H2, poles, params2);
        end
        [V1, KK1, HH1, ~] = rat_krylov(DA1, VV1, K1, H1, inf, params1); % artificial step at infinity to get the projected matrices 
        [V2, KK2, HH2, ~] = rat_krylov(DA2, VV2, K2, H2, inf, params2);
        
        if (mod(j, lag) == 1 || lag == 1)
            % Check that the subspaces are not too large compared to the
            % size of the current block. If this is the case
            % then we do the computation for full matrices
            if (size(V1, 2) > size(A, 1)/4 || size(V2, 2) > size(A, 1)/4)
                warning("Switched to dense computation because dimension of Krylov subspace is too large");
                X = hss(f(full(A)));
                return
            end

            % Compute the projected matrices TODO: handle the case of ill conditioned matrices K with deflation 
            if cond(KK1(1:end-k, :)) < 1e10 % was 15
                AA1 = HH1(1:end-k, :) / KK1(1:end-k, :); 
            else
                warning('HSS_FUN_DAC:: ill-conditioned matrix K, projection computed with additional matvecs')
                AA1 = V1(:, 1:end-k)' * DA1 * V1(:, 1:end-k);
            end
            if cond(KK1(1:end-k, :)) < 1e10 % was 15
                AA2 = HH2(1:end-k, :) / KK2(1:end-k, :); 
            else
                warning('HSS_FUN_DAC:: ill-conditioned matrix K, projection computed with additional matvecs')
                AA2 = V2(:, 1:end-k)' * DA2 * V2(:, 1:end-k);
            end

            UU = blkdiag(V1(:, 1:end-1)' * Z, V2(:, 1:end-1)' * W);

            % enforce symmetry
            AA1 = (AA1 + AA1') / 2;
            AA2 = (AA2 + AA2') / 2;
            tAA = blkdiag(AA1, AA2) + UU * B * UU';
            tAA = (tAA + tAA') / 2;
            DF = f(tAA) - blkdiag(f(AA1), f(AA2));

            if j > lag % check the stopping criterion
                q = (size(DF, 1) - size(DFold, 1))/2;
                h = size(DFold, 1)/2;
                % We need to reorder DFold because of the block diagonal form of the Krylov space
                DFold = blkdiag(DFold, zeros(2 * q));
                ind = [1 : h, 2 * h + 1 : 2 * h + q, h + 1 : 2 * h, 2 * h + q + 1 : 2 * h + 2 * q];
                DFold = DFold(ind, ind);
                err = norm(DF - DFold, 'fro');

    %------------------------DEBUG--------------------------------------------------------------------------------------------
                if debug == 1
                    JJ = blkdiag(V1(:, 1:end - 1), V2(:, 1:end-1));
                    approxDF = JJ * DF * JJ';
                    orthproj = trueDF - JJ * (JJ' * trueDF);
                    fprintf('j = %d, true error = %1.2e, estimated error = %1.2e, orth. projection error = %1.2e, orthonormality = %1.2e\n', j, norm(trueDF - approxDF, 'fro'), err, norm(orthproj, 'fro'), norm(JJ'*JJ-eye(size(JJ,2))));
                end
                if debug == 2 || debug == 3
                    fprintf('j = %d, estimated error = %1.2e\n', j, err);
                end
    %-----------------------END DEBUG----------------------------------------------------------------------------------------

                if err < tol
                    if (debug == 3)
                        fprintf("Used a %d-dimensional update\n", size(DF, 1));
                    end
                    break
                end
            end
            DFold = DF;
        end

    end

    V = blkdiag(V1(:, 1:end - 1), V2(:, 1:end-1));

    % Compress correction
    [E, S, F] = svd(DF);
    r = sum(diag(S) > tol);
    DF = S(1:r, 1:r);
    U = V * E(:, 1:r);
    V = V * F(:, 1:r);

%------------------------DEBUG--------------------------------------------------------------------------------------------	
	if debug == 1
		rk = sum(svd(trueDF)>tol);
		fprintf('final space dimension = %d, maxit = %d, rank of the true DF = %d\n', size(V, 2), maxit, rk);
	end
	if debug == 2
		fprintf('final space dimension = %d, maxit = %d\n', size(V, 2), maxit);
	end
%------------------------END DEBUG----------------------------------------------------------------------------------------

	if j == maxit 
		warning('HSS_FUN_DAC_INVSQRT::maxit iteration reached, update has been computed with dense arithmetic at size %d', size(A, 1))
 		V = eye(n);
		fA = full(A); fDA = full(DA);
		fA = (fA + fA') / 2; fDA = (fDA + fDA') / 2; 
		DF = f(fA) - f(fDA);
    end
	X = X + hss('low-rank', U, V, DF);
end
