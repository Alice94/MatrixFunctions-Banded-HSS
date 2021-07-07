function d = compute_diag(A, normfA, debug, minblocksize, onlytrace, tol, mmq, lag)
    f = @expm;

    if (size(A, 1) <= minblocksize)
        d = do_dense_computations(A, onlytrace, tol, mmq);
        return;
    end

    options.ufactor = 100;
    [map, ~] = metismex('PartGraphRecursive', A, 2, options);
    I1 = [];
    I2 = [];
    for h = 1:size(A, 1)
        if (map(h) == 0)
            I1 = [I1, h];
        else
            I2 = [I2, h];
        end
    end
    
    n1 = length(I1);
    n2 = length(I2);
    
    if (debug >= 2)
        fprintf("Size of partitions: %d and %d\n", n1, n2);
    end

    % Compute the rank of the off-diagonal correction
    B = A(I2, I1);
    [I, J, ~] = find(B);
    I = unique(I);
    J = unique(J);
    C = B(I,J);
    [UU, SS, VV] = svd(full(C));
    if (size(SS, 1) == 1 || size(SS, 2) == 1)
        if (SS(1,1) > tol)
            r = 1;
        else
            r = 0;
        end
    else
        r = sum(diag(SS) > tol);
    end
    
    % If rank is too large, do dense computations
    if (r > min(n1, n2)/15)
        if (debug >= 0)
            fprintf("Off-diagonal rank of block of size %d is too large, doing dense computations\n", n1+n2);
        end
        d = do_dense_computations(A, onlytrace, tol, mmq);
        return;
    end
    
    UU = UU(:, 1:r);
    SS = SS(1:r, 1:r); % added this
    VV = VV(:, 1:r);
    if (debug > 0)
        fprintf("Blocks have sizes %d and %d, off-diagonal part has rank 2 * %d\n", length(I1), length(I2), r)
    end
    DA1 = A(I1, I1);
    DA2 = A(I2, I2);
    
    if (r ~= 0)
        W = sparse(n2, r);
        W(I, :) = UU;
        Z = sparse(n1, r);
        Z(J, :) = VV;

        %------------------------DEBUG--------------------------------------------------------------------------------------------
        if debug == 1
            fprintf('--------- block size: %d, cond(A) = %1.2e, sought accuracy = %1.2e ---------\n', size(A,1), cond(full(A)), tol)
            fA = full(A([I1, I2], [I1, I2])); 
            fDA1 = full(DA1); fDA2 = full(DA2);
            fDA1 = (fDA1 + fDA1') / 2; 	fDA2 = (fDA2 + fDA2') / 2; fA = (fA + fA') / 2;
            trueDF = f(fA) - blkdiag(f(fDA1), f(fDA2));
        end
        if debug == 2
            fprintf('--------- block size: %d, sought accuracy = %1.2e ---------\n', size(A,1), tol)
        end
    %-----------------------END DEBUG----------------------------------------------------------------------------------------



        
        % Compute the correction with a Krylov method
        maxit = floor(min(n1, n2) / (2*r));
        k1 = size(Z, 2);
        k2 = size(W, 2);
        kk1 = k1;
        kk2 = k2;
        for j = 1:maxit
            % Polynomial Krylov
            if j == 1
                [V1, H1, params1, ~] = poly_krylov(DA1, Z);
                [V2, H2, params2, ~] = poly_krylov(DA2, W);
                % Recompute SS because poly_krylov might change the bases Z
                % and W
                SS = V2(:, 1:r)' * B * V1(:, 1:r);
            else
                [V1, H1, params1, ~] = poly_krylov(V1, H1, params1);
                [V2, H2, params2, ~] = poly_krylov(V2, H2, params2);
            end

            k1 = size(params1.last, 2);
            k2 = size(params2.last, 2);
            kk1 = [kk1, k1];
            kk2 = [kk2, k2];
            
            if (mod(j, lag) == 1 || lag == 1)
                % Check that the subspaces are not too large compared to the
                % size of the current block. If this is the case
                % then we do the computation for full matrices
                if (size(V1, 2) > min(n1, n2)/2 || size(V2, 2) > min(n1, n2)/2)
                    warning("Switched to dense computation because dimension of Krylov subspace is too large");
                    d = do_dense_computations(A, onlytrace, tol, mmq);
                    return
                end
                
                AA1 = H1(1:end-k1, :); 
                AA2 = H2(1:end-k2, :); 
                
                % enforce symmetry
                AA1 = (AA1 + AA1') / 2;
                AA2 = (AA2 + AA2') / 2;
                tAA = blkdiag(AA1, AA2) / 2;
                tAA(1:r, size(AA1, 1)+1 : size(AA1, 1)+r) = spdiags(diag(SS), 0, r, r);
                tAA = tAA + tAA';
                
%                 % THEN REMOVE
%                 JJ = blkdiag(V1(:, 1:end - k1), V2(:, 1:end-k2));
%                 tAA = JJ' * A([I1, I2], [I1, I2]) * JJ;
%                 % END REMOVE

                if onlytrace == 0 % diagonal needed
                    DF = expm(tAA) - blkdiag(expm(AA1), expm(AA2));
                else 
                    eig1 = eig(full(tAA));
                    eig2 = eig(full(AA1));
                    eig3 = eig(full(AA2));
                    DF = sum(exp(eig1)) - sum(exp(eig2)) - sum(exp(eig3));
                end

                if j > lag % check the stopping criterion
                    if onlytrace == 0
                        h1 = sum(kk1(1:end-1-lag));
                        h2 = sum(kk2(1:end-1-lag));
                        l1 = sum(kk1(end-lag:end-1));
                        l2 = sum(kk2(end-lag:end-1));
                        % We need to reorder DFold because of the block diagonal form of the Krylov space
                        DFold = blkdiag(DFold, zeros(l1+l2));
                        ind = [1:h1, h1+h2+1 : h1+h2+l1, h1+1:h1+h2, h1+h2+l1+1:h1+h2+l1+l2];
                        DFold = DFold(ind, ind);
                    end
                    err = norm(DF - DFold, 'fro');


            %------------------------DEBUG--------------------------------------------------------------------------------------------
                    if debug == 1
                        JJ = blkdiag(V1(:, 1:end - k1), V2(:, 1:end-k2));
                        approxDF = JJ * DF * JJ';
                        orthproj = trueDF - JJ * (JJ' * trueDF);
                        fprintf('Size of V1 is %d and size of V2 is %d \n', size(V1, 2), size(V2, 2))
                        fprintf('j = %d, true error = %1.2e, estimated error = %1.2e, orth. projection error = %1.2e, orthonormality = %1.2e\n',...
                            j, norm(trueDF - approxDF, 'fro'), err, norm(orthproj, 'fro'), norm(JJ'*JJ-eye(size(JJ,2))));
                    end
                    if debug == 2
                        fprintf('j = %d, estimated error = %1.2e\n', j, err);
                    end
            %-----------------------END DEBUG----------------------------------------------------------------------------------------

                    if err < tol * normfA
                        if (debug >= 2)
                            fprintf("Used a %d-dimensional update\n", size(DF, 1));
                        end
                        break
                    end
                end
                DFold = DF;
            end
        end

        V = blkdiag(V1(:, 1:end - k1), V2(:, 1:end-k2));

        if j == maxit 
            warning('Maxit iteration reached, update has been computed with dense arithmetic at size %d', size(A, 1))
            d = do_dense_computations(A, onlytrace, tol, mmq);
            return;
        end

        d1 = compute_diag(DA1,normfA,debug,minblocksize,onlytrace,tol,mmq,lag);
        d2 = compute_diag(DA2,normfA,debug,minblocksize,onlytrace,tol,mmq,lag);
        if (onlytrace == 1)
            d = d1 + d2 + DF;
        else
            d = [d1; d2];
            
            % Compress correction
            [Q, S] = eig(DF, 'vector');
            ind = find(abs(S) > tol * normfA);
            DF = diag(S(ind));
            V = V * Q(:, ind);  
            
            U = V * DF;
            V = U .* V;
            d = d + sum(V, 2);
            
            % Need to re-sort stuff
            invperm([I1, I2]) = 1:size(A, 1);
            d = d(invperm);
        end
    else
        % very lucky case, the matrix is block diagonal!
        d1 = compute_diag(DA1,normfA,debug,minblocksize,onlytrace,tol,mmq,lag);
        d2 = compute_diag(DA2,normfA,debug,minblocksize,onlytrace,tol,mmq,lag);
        if (onlytrace == 1)
            d = d1 + d2;
        else
            d = [d1; d2];
            % Need to re-sort stuff
            invperm([I1, I2]) = 1:size(A, 1);
            d = d(invperm);
        end
    end

end



function d = do_dense_computations(A, onlytrace, tol, mmq)
    if (size(A, 1) > 10000)
        warning("matrix A is too large");
        fprintf("Matrix A is too large to carry out dense computations\n");
        if (onlytrace == 1)
            d = 0;
            return;
        else
            d = zeros(size(A, 1), 1);
            return;
        end
    end
    if (onlytrace == 1)
        d = sum(exp(eig(full(A))));
    else
        n = size(A, 1);
        if (mmq == 0)
            d = diag(expm(A));
        else
            % Using mmq toolbox (only for exponential):
            d = zeros(n, 1);
            for i = 1:n
                [d(i), ~] = mmq_exp_given_tol(A, i, tol);
            end
        end
    end
end







