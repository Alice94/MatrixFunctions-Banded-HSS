function X = hss_fun_dac(A, f, poles, debug, normfA, isherm, onlyd)
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
	if ~exist('isherm', 'var')
		isherm = issymmetric(A);
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
	if size(A, 1) <= hssoption('block-size') || ~isempty(A.D)
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

	DA1 = A.A11; DA1.topnode = 1; 
	DA2 = A.A22; DA2.topnode = 1; 
	DA = blkdiag(DA1, DA2);

	% Recursive call on the diagonal blocks
	if onlyd == 0
		X = blkdiag(hss_fun_dac(DA1, f, poles, debug, normfA, isherm, onlyd), hss_fun_dac(DA2, f, poles, debug, normfA, isherm, onlyd)); % function of the diagonal blocks
	elseif onlyd == 1
		X = [hss_fun_dac(DA1, f, poles, debug, normfA, isherm, onlyd); hss_fun_dac(DA2, f, poles, debug, normfA, isherm, onlyd)];
	elseif onlyd == 2
		X = hss_fun_dac(DA1, f, poles, debug, normfA, isherm, onlyd) + hss_fun_dac(DA2, f, poles, debug, normfA, isherm, onlyd);
	else
		error('HSS_FUN_DAC:: unsupported value of onlyd')
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
		if isherm
			fDA1 = (fDA1 + fDA1') / 2; 	fDA2 = (fDA2 + fDA2') / 2; fA = (fA + fA') / 2;
		end
		trueDF = f(fA) - blkdiag(f(fDA1), f(fDA2));
	end
	if debug == 2
		fprintf('--------- block size: %d, sought accuracy = %1.2e ---------\n', n, tol)
	end
%-----------------------END DEBUG----------------------------------------------------------------------------------------

	lag = 1; % lag parameter for the stopping criterion of the Krylov method
	DFold = {};
	if isherm % Hermitian case	
			
		% Retrieve the low-rank factorization of the offdiagonal part
		[W, Z] = offdiag(A, 'lower'); 
		k = size(W, 2);
		if k == 0 % if the offdiagonal part is zero, then there is nothing else to compute
			return
		end
		[W, S, Z] = compress_factors(W, Z, normfA);
		k = size(W, 2);
		if k == 0 % if the offdiagonal part is negligible, then there is nothing else to compute
			return
		end
		B = [zeros(k), S; S', zeros(k)];	
		
		% Compute the correction with a Krylov method
		maxit = ceil( (n - k) / (length(poles) * k));
		for j = 1:maxit
			if length(poles) == 1 && poles == inf % polynomial Krylov
				if j == 1
					[V1, HH1, params1] = poly_krylov(DA1, Z);
					[V2, HH2, params2] = poly_krylov(DA2, W);
				else
					[V1, HH1, params1] = poly_krylov(V1, HH1, params1);
					[V2, HH2, params2] = poly_krylov(V2, HH2, params2);
				end
			elseif length(poles) == 2 && (prod(poles == [0 inf]) || prod(poles == [inf 0])) % extended Krylov method
				if j == 1
					if like_extended
						DA1 = ek_struct(DA1- sh * hss('eye', size(DA1, 1))); DA2 = ek_struct(DA2- sh * hss('eye', size(DA2, 1)));
						[V1, KK1, HH1, params1] = ek_krylov(DA1, Z);
						[V2, KK2, HH2, params2] = ek_krylov(DA2, W);
					else
						DA1 = ek_struct(DA1); DA2 = ek_struct(DA2);
						[V1, KK1, HH1, params1] = ek_krylov(DA1, Z);
						[V2, KK2, HH2, params2] = ek_krylov(DA2, W);
					end
				else
					[V1, KK1, HH1, params1] = ek_krylov(V1, KK1, HH1, params1);
					[V2, KK2, HH2, params2] = ek_krylov(V2, KK2, HH2, params2);
				end
			else % Rational Krylov that repeats cyclically the poles provided by the user
				if j == 1
					%DA1 = rk_struct(DA1); DA2 = rk_struct(DA2);
					%keyboard
					[VV1, K1, H1, params1] = rk_krylov(DA1, Z, poles);
					[VV2, K2, H2, params2] = rk_krylov(DA2, W, poles);
				else
					[VV1, K1, H1, params1] = rk_krylov(DA1, VV1, K1, H1, poles, params1);
					[VV2, K2, H2, params2] = rk_krylov(DA2, VV2, K2, H2, poles, params2);
				end
				[V1, KK1, HH1, ~] = rk_krylov(DA1, VV1, K1, H1, inf, params1); % artificial step at infinity to get the projected matrices 
				[V2, KK2, HH2, ~] = rk_krylov(DA2, VV2, K2, H2, inf, params2);
            end	
            
            % Check that the subspaces are not too large compared to the
            % size of the current block. If this is the case
            % then we do the computation for full matrices
            if (size(V1, 2) > size(A, 1)/4 || size(V2, 2) > size(A, 1)/4)
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
                n1 = size(HH1, 2); n2 = size(HH2, 2);
				AA1 = HH1(1:n1,:); AA2 = HH2(1:n2,:);
			else
                if cond(KK1(1:end-k, :)) < 1e15
                    AA1 = HH1(1:end-k, :) / KK1(1:end-k, :); 
                else
                    warning('HSS_FUN_DAC:: ill-conditioned matrix K, projection computed with additional matvecs')
                    if (isa(DA1, "hss"))
                        AA1 = V1(:, 1:end-k)' * DA1 * V1(:, 1:end-k);
                    else
                        AA1 = V1(:, 1:end-k)' * DA1.multiply(1.0, 0.0, V1(:, 1:end-k));
                    end
                end
                if cond(KK1(1:end-k, :)) < 1e15
                    AA2 = HH2(1:end-k, :) / KK2(1:end-k, :); 
                else
                    warning('HSS_FUN_DAC:: ill-conditioned matrix K, projection computed with additional matvecs')
                    if (isa(DA2, "hss"))
                        AA2 = V2(:, 1:end-k)' * DA2 * V2(:, 1:end-k);
                    else
                        AA2 = V2(:, 1:end-k)' * DA2.multiply(1.0, 0.0, V2(:, 1:end-k));
                    end
                end
            end

			if like_extended
				AA1 = AA1 + sh * eye(size(AA1, 1));	AA2 = AA2 + sh * eye(size(AA2, 1));
			end
			UU = blkdiag(V1(:, 1:end-k)' * Z, V2(:, 1:end-k)' * W);

			% enforce symmetry
			AA1 = (AA1 + AA1') / 2;
			AA2 = (AA2 + AA2') / 2;
			%AA = (AA + AA') / 2;
			tAA = blkdiag(AA1, AA2) + UU * B * UU';
			%tAA = AA + UU * B * UU';
			tAA = (tAA + tAA') / 2;
			if onlyd == 0 || onlyd == 1
				DF = f(tAA) - blkdiag(f(AA1), f(AA2));
			else % in the case of the trace
				eig1 = eig(tAA);
				eig2 = [eig(AA1); eig(AA2)];
				DF = sum(f(eig1) - f(eig2));
			end

			if j > lag % check the stopping criterion
				q = (size(DF, 1) - size(DFold{1}, 1))/2;
				h = size(DFold{1}, 1)/2;
				% We need to reorder DFold because of the block diagonal form of the Krylov space
				DFold{1} = blkdiag(DFold{1}, zeros(2 * q));
				ind = [1 : h, 2 * h + 1 : 2 * h + q, h + 1 : 2 * h, 2 * h + q + 1 : 2 * h + 2 * q];
				DFold{1} = DFold{1}(ind, ind);
				err = norm(DF - DFold{1}, 'fro');

%------------------------DEBUG--------------------------------------------------------------------------------------------
				if debug == 1
					JJ = blkdiag(V1(:, 1:end - k), V2(:, 1:end-k));
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

		V = blkdiag(V1(:, 1:end - k), V2(:, 1:end-k));

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
	else % non Hermitian case
		
		% Retrieve the low-rank factorization of the off-diagonal part
		[W, Z] = offdiag(A, 'all'); 
		k = size(W, 2);	
		if k == 0 % if the offdiagonal part is zero, then there is nothing else to compute
			return
		end
		[W, S, Z] = compress_factors(W, Z, normfA);
		k = size(W, 2);	
		if k == 0 % if the offdiagonal part is negligible, then there is nothing else to compute
			return
		end

		% Compute the correction with a Krylov method
		maxit = ceil( (n - k) / (length(poles) * k));
		for j = 1:maxit
			if length(poles) == 1 && poles == inf % polynomial Krylov
				if j == 1
					[V1, HH1, params1] = poly_krylov(DA, W);
					[V2, HH2, params2] = poly_krylov(DA', Z);
				else
					[V1, HH1, params1] = poly_krylov(V1, HH1, params1);
					[V2, HH2, params2] = poly_krylov(V2, HH2, params2);
				end
			elseif length(poles) == 2 && (prod(poles == [0 inf]) || prod(poles == [inf 0])) % extended Krylov method
				if j == 1
					[V1, KK1, HH1, params1] = ek_krylov(DA, W);
					[V2, KK2, HH2, params2] = ek_krylov(DA', Z);
				else
					[V1, KK1, HH1, params1] = ek_krylov(V1, KK1, HH1, params1);
					[V2, KK2, HH2, params2] = ek_krylov(V2, KK2, HH2, params2);
				end
			else % Rational Krylov that repeats cyclically the poles provided by the user
				if j == 1
					%[DA1, DA2] = rk_struct(DA); DA2 = rk_struct(DA2);
					[VV1, K1, H1, params1] = rk_krylov(DA, Z, poles);
					[VV2, K2, H2, params2] = rk_krylov(DA', W, poles);
				else
					[VV1, K1, H1, params1] = rk_krylov(DA, VV1, K1, H1, poles, params1);
					[VV2, K2, H2, params2] = rk_krylov(DA', VV2, K2, H2, poles, params2);
				end
				[V1, KK1, HH1, ~] = rk_krylov(DA1, VV1, K1, H1, inf, params1); % artificial step at infinity to get the projected matrices 
				[V2, KK2, HH2, ~] = rk_krylov(DA2, VV2, K2, H2, inf, params2);
			end
			UU = V1(:, 1:end-k)' * W; UU2 = V2(:, 1:end-k)' * W;
			VV = V2(:, 1:end-k)' * Z;
            
            % Check that the subspaces are not too large compared to the
            % size of the current block. If this is the case
            % then we do the computation for full matrices
            if (size(V1, 2) > size(DA, 1)/2 || size(V2, 2) > size(DA, 1)/2)
                warning("Switched to dense computation because dimension of Krylov subspace is too large");
                if onlyd == 0
                    X = hss(f(full(A)));
%                     X.D = f(full(A));
%                     X.topnode = 1;
%                     X.leafnode = 1;
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

			if length(poles) == 1 && poles == inf % if it is polynomial Krylov
                n1 = size(HH1, 2); n2 = size(HH2, 2);
				AA1 = HH1(1:n1,:); AA2 = HH2(1:n2,:);
			else
				% Compute the projected matrices TODO: handle the case off ill conditioned matrices K with deflation 
				if  cond(KK1(1:end-k, :)) < 1e15
					AA1 = HH1(1:end-k, :) / KK1(1:end-k, :); 
				else
					warning('HSS_FUN_DAC:: ill-conditioned matrix K, projection computed with additional matvecs')
		            if (isa(DA1, "hss"))
		                AA1 = V1(:, 1:end-k)' * DA1 * V1(:, 1:end-k);
		            else
		                AA1 = V1(:, 1:end-k)' * DA1.multiply(1.0, 0.0, V1(:, 1:end-k));
		            end
				end
				if cond(KK2(1:end-k, :)) < 1e15
					AA2 = HH2(1:end-k, :) / KK2(1:end-k, :); 
				else
					warning('HSS_FUN_DAC:: ill-conditioned matrix K, projection computed with additional matvecs')
		            if (isa(DA2, "hss"))
		                AA2 = V2(:, 1:end-k)' * DA2 * V2(:, 1:end-k);
		            else
		                AA2 = V2(:, 1:end-k)' * DA2.multiply(1.0, 0.0, V2(:, 1:end-k));
		            end
				end
			end

			AA = blkdiag(AA1, AA2' + UU2 * S * VV'); % projected matrix
			AA(1:size(UU, 1), end - size(VV, 1) + 1:end) = UU * S * VV';
			DF = f(AA); DF = DF(1:size(UU, 1), end - size(VV, 1) + 1:end);

			if j > lag % check the stopping criterion
				nn = size(DF, 1);
				DFold{1}(nn, nn) = 0;
				err = norm(DF - DFold{1}, 'fro');

%---------------------------DEBUG----------------------------------------------------------------------------------------
				if debug == 1
					approxDF = V1(:, 1:end - k) * DF * V2(:, 1:end - k)';

					fprintf('j = %d, true error = %1.2e, estimated error = %1.2e\n', j, norm(trueDF - approxDF), err);
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
		U = V1(:, 1:end-k);
		V = V2(:, 1:end-k);

		if onlyd == 2 % in the case of the trace
			X = X + trace(V' * U * DF);
			return
		end

		if onlyd == 1 % in the case of the diagonal
			for j = 1:size(V, 1)
				X(j) = X(j) + U(j, :) * DF * V(j, :)';
			end
			return
		end

		% Compress correction
		[E, S, F] = svd(DF);
		r = sum(diag(S) > tol);
		DF = S(1:r, 1:r);
		U = U * E(:, 1:r);
		V = V * F(:, 1:r);
	end

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
		if isherm
			fA = (fA + fA') / 2; fDA = (fDA + fDA') / 2; 
		end
		DF = f(fA) - f(fDA);
	end
	if isherm % Sum the low-rank update to the block diagonal part
        X = X + hss('low-rank', V, V, DF);
	else
		X = X + hss('low-rank', U, V, DF);
	end
end
