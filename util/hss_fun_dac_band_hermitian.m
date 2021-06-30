function X = hss_fun_dac_band_hermitian(A, f, poles, debug, normfA, onlyd)
% Computes f(A) in the HSS format for an HSS A, or diag(f(A)), or trace(f(A))
%----------------------------INPUT-----------------------------------------------------------------------------------
%
% A			banded matrix
% f			handle function that computes the function on a matrix argument (e.g. @expm)
%           In the case of the trace, f should apply entrywise to a vector argument
% poles		shift parameters for the Krylov method used for computing the update
% debug		allows some debugging print
% normfA	estimate of the norm of fA 
% onlyd  	flag that denotes whether only the diagonal (onlyd = 1) or the trace (onlyd = 2) is computed. 
%			
%
%----------------------------OUTPUT----------------------------------------------------------------------------------
%
% X			if onlyd = 0 is the HSS representation of f(A); otherwise its diagonal (onlyd = 1) or trace (onlyd = 2)
%
%--------------------------------------------------------------------------------------------------------------------
	if ~exist('poles', 'var') 
		poles = [inf 0];
	end
	if prod(poles == inf)
		poles = inf;
	end 
	if ~exist('debug', 'var')
		debug = 0;
	end
	if ~exist('normfA', 'var')
		normfA = 1;
    end
	if ~exist('onlyd', 'var') || isempty(onlyd)
		onlyd = 0;
	end

	tol = normfA * hssoption('threshold'); % set relative accuracy

	[m, n] = size(A); 
	if m ~= n
		error('HSS_FUN_DAC:: the input argument is not a square matrix');
	end

	% Base case of the recursion
	if size(A, 1) <= hssoption('block-size')
		if onlyd == 0
			X = hss();
			X.D = f(full(A));
			X.topnode = 1;
			X.leafnode = 1;
			return
		elseif onlyd == 1
			X = diag(f(full(A)));
			return
		elseif onlyd == 2
			eigen = eig(full(A));
			X = sum(f(eigen));
			return
		else
			error('HSS_FUN_DAC:: unsupported value of onlyd')
		end
    end
    
    n1 = ceil(n/2);
    n2 = floor(n/2);
    DA1 = A(1:n1, 1:n1);
    DA2 = A(n1+1:end, n1+1:end);
    
    % Retrieve a low-rank factorization of the offdiagonal part and update
    % diagonal blocks
    b = bandwidth(A);
    [U1, S1, V1] = svd(full(A(n1+1:n1+b, n1-b+1:n1)));
    if (size(S1, 1) == 1 || size(S1, 2) == 1)
        if (S1(1,1) > tol)
            k = 1;
        else
            k = 0;
        end
    else
        k = sum(diag(S1) > tol);
    end
    if (k > 0)
        if (k < b)
            U1 = U1(:, 1:k);
            V1 = V1(:, 1:k);
            S1 = S1(1:k, 1:k);
        end
        if (k == 1)
            U2 = [U1; sparse(n2-b, 1)];
            V2 = [sparse(n1-b, 1); V1];
        else
            U2 = [U1; sparse(n2-b, k)];
            V2 = [sparse(n1-b, k); V1];
        end
        W = full([V2; -U2]);
        B = full(-S1);
        DA1(n1-b+1:n1, n1-b+1:n1) = DA1(n1-b+1:n1, n1-b+1:n1) + V1*S1*V1';
        DA2(1:b, 1:b) = DA2(1:b, 1:b) + U1*S1*U1';
        DA1 = (DA1 + DA1')/2; DA2 = (DA2 + DA2')/2;
    end
    
    DA = blkdiag(DA1, DA2);

	% Recursive call on the diagonal blocks
	if onlyd == 0
		X = blkdiag(hss_fun_dac_band_hermitian(DA1, f, poles, debug, normfA, onlyd), hss_fun_dac_band_hermitian(DA2, f, poles, debug, normfA, onlyd)); % function of the diagonal blocks
	elseif onlyd == 1
		X = [hss_fun_dac_band_hermitian(DA1, f, poles, debug, normfA, onlyd); hss_fun_dac_band_hermitian(DA2, f, poles, debug, normfA, onlyd)];
	elseif onlyd == 2
		X = hss_fun_dac_band_hermitian(DA1, f, poles, debug, normfA, onlyd) + hss_fun_dac_band_hermitian(DA2, f, poles, debug, normfA, onlyd);
	else
		error('HSS_FUN_DAC:: unsupported value of onlyd')
    end
    
    if k == 0 % if the offdiagonal part is negligible, then there is nothing else to compute
        return
    end

	like_extended = 0; % flag that will indicate whether we have two poles with one of the two equal to 'inf'
	if length(poles) == 2 && ~(prod(poles == [0 inf]) || prod(poles == [inf 0])) && (poles(1) == inf || poles(2) == inf) 
		like_extended = 1;
		sh = poles(find(poles < inf));
		poles = [0 inf];
	end

%------------------------DEBUG--------------------------------------------------------------------------------------------
	if debug == 1
		fprintf('--------- block size: %d, cond(A) = %1.2e, sought accuracy = %1.2e ---------\n', n, cond(full(A)), tol)
		fA = full(A); 
		fDA1 = full(DA1); fDA2 = full(DA2);
		fDA1 = (fDA1 + fDA1') / 2; 	fDA2 = (fDA2 + fDA2') / 2; fA = (fA + fA') / 2;
		trueDF = f(fA) - blkdiag(f(fDA1), f(fDA2));
	end
	if debug == 2
		fprintf('--------- block size: %d, sought accuracy = %1.2e ---------\n', n, tol)
	end
%-----------------------END DEBUG----------------------------------------------------------------------------------------

	lag = 1; % lag parameter for the stopping criterion of the Krylov method
	DFold = {};			

    % Compute the correction with a Krylov method
    maxit = ceil( (n - k) / (length(poles) * k));
    for j = 1:maxit
        if length(poles) == 1 && poles == inf % polynomial Krylov
            if j == 1
                [V1, HH1, params1] = poly_krylov(DA, W);
            else
                [V1, HH1, params1] = poly_krylov(V1, HH1, params1);
            end
        elseif length(poles) == 2 && (prod(poles == [0 inf]) || prod(poles == [inf 0])) % extended Krylov method
            if j == 1
                if like_extended
                    DA = ek_struct(DA- sh * hss('eye', size(DA, 1))); 
                    [V1, KK1, HH1, params1] = ek_krylov(DA, W);
                else
                    DA = ek_struct(DA);
                    [V1, KK1, HH1, params1] = ek_krylov(DA, W);
                end
            else
                [V1, KK1, HH1, params1] = ek_krylov(V1, KK1, HH1, params1);
            end
        else % Rational Krylov that repeats cyclically the poles provided by the user
            if j == 1
                [VV1, K1, H1, params1] = rk_krylov(DA, W, poles);
            else
                [VV1, K1, H1, params1] = rk_krylov(DA, VV1, K1, H1, poles, params1);
            end
            [V1, KK1, HH1, ~] = rk_krylov(DA, VV1, K1, H1, inf, params1); 
            % artificial step at infinity to get the projected matrices 
        end	

        % Check that the subspaces are not too large compared to the
        % size of the current block. If this is the case
        % then we do the computation for full matrices
        if size(V1, 2) > size(A, 1)/2 
            warning("Switched to dense computation because dimension of Krylov subspace is too large");
            if onlyd == 0
                X = hss(f(full(A)));
                return
            elseif onlyd == 1
                X = diag(f(full(A)));
                return
            elseif onlyd == 2
                eigen = eig(full(A));
                X = sum(f(eigen));
                return
            else
                error('HSS_FUN_DAC:: unsupported value of onlyd')
            end
        end

        % Compute the projected matrices TODO: handle the case of ill conditioned matrices K with deflation 
        if length(poles) == 1 && poles == inf % if it is polynomial Krylov
            AA1 = HH1(1:size(HH1, 2),:);
        else
            if cond(KK1(1:end-k, :)) < 1e15
                AA1 = HH1(1:end-k, :) / KK1(1:end-k, :); 
            else
                warning('HSS_FUN_DAC:: ill-conditioned matrix K, projection computed with additional matvecs')
                AA1 = V1(:, 1:end-k)' * DA * V1(:, 1:end-k);
            end
        end

        if like_extended
            AA1 = AA1 + sh * eye(size(AA1, 1));
        end
        UU = V1(:, 1:end-k)' * W;

        % enforce symmetry
        AA1 = (AA1 + AA1') / 2;
        tAA = AA1 + UU * B * UU';
        tAA = (tAA + tAA') / 2;
        if onlyd == 0 || onlyd == 1
            DF = f(tAA) - f(AA1);
        else % in the case of the trace
            eig1 = eig(tAA);
            eig2 = eig(AA1);
            DF = sum(f(eig1) - f(eig2));
        end

        if j > lag % check the stopping criterion
            q = (size(DF, 1) - size(DFold{1}, 1));
            DFold{1} = blkdiag(DFold{1}, zeros(q));
            err = norm(DF - DFold{1}, 'fro');

%------------------------DEBUG--------------------------------------------------------------------------------------------
            if debug == 1
                JJ = V1(:, 1:end - k);
                approxDF = JJ * DF * JJ';
                orthproj = trueDF - JJ * (JJ' * trueDF);
                if onlyd == 0 || onlyd == 1
                    fprintf('j = %d, true error = %1.2e, estimated error = %1.2e, orth. projection error = %1.2e, orthonormality = %1.2e\n', j, norm(trueDF - approxDF, 'fro'), err, norm(orthproj, 'fro'), norm(JJ'*JJ-eye(size(JJ,2))));
                else
                    fprintf('j = %d, true error = %1.2e, estimated error = %1.2e, orth. projection error = %1.2e, orthonormality = %1.2e\n', j, abs(trace(trueDF - approxDF)), err, abs(trace(orthproj)), norm(JJ'*JJ-eye(size(JJ,2))));
                end
            end
            if debug == 2
                fprintf('j = %d, estimated error = %1.2e\n', j, err);
            end
%-----------------------END DEBUG----------------------------------------------------------------------------------------

            if err < tol
                break
            end
            DFold = [DFold(2:end), DF];
        else
            DFold = [DFold, DF];
        end

    end
    if onlyd == 2 % in the case of the trace
        X = X + DF;
        return
    end

    V = V1(:, 1:end - k);

    if onlyd == 1 % in the case of the diagonal
        for j = 1:size(V, 1)
            X(j) = X(j) + V(j, :) * DF * V(j, :)';
        end
        return
    end

    % Compress correction
    [Q, S] = eig(DF, 'vector');
    ind = find(abs(S) > tol);
    DF = diag(S(ind));
    V = V * Q(:, ind);

%------------------------DEBUG--------------------------------------------------------------------------------------------	
	if debug == 1
		rk = sum(svd(trueDF)>tol);
		fprintf('final space dimesion = %d, maxit = %d, rank of the true DF = %d\n', size(V, 2), maxit, rk);
	end
	if debug == 2
		fprintf('final space dimesion = %d, maxit = %d\n', size(V, 2), maxit);
	end
%------------------------END DEBUG----------------------------------------------------------------------------------------

	if j == maxit 
		warning('HSS_FUN_DAC::maxit iteration reached, update has been computed with dense arithmetic at size %d', size(A, 1))
 		V = eye(n);
		fA = full(A); fDA = full(DA);
		fA = (fA + fA') / 2; fDA = (fDA + fDA') / 2; 
		DF = f(fA) - f(fDA);
    end
	X = X + hss('low-rank', V, V, DF);
end
