function F = FBandedPolyAdaptive(A, b, f, minbs, tol, debug)
    % function [F] = FBandedPolyAdaptive(A, b, f, initial_block_size, tol, debug)
    % Computes a banded approximation of a banded matrix A using polynomial
    % Krylov subspaces

    lag = 1;

    % Check that initial_block_size is sufficiently large
    if (minbs < 4*b)
        minbs = 4*b;
    elseif (mod (minbs, 2) == 1)
        minbs = minbs + 1;
    end

    n = size(A, 1);
    if (minbs * 2 > n)
        F = f(A);
    else
        % Compute diagonal blocks
        cont = 1;
        fDiagBlocks = {};
        dimDiagBlocks = [];
        splittingPoints = [];
        fSmallerBlocks = {};
        bandwidthBlocks = []; % the last one could be large

        i = 1;
        bs = minbs;
        while (i <= n) 
            % A new diagonal block is needed
            bs = max(bs/2, minbs);
            if (bs > (n-i+1)/2)
                bs = n-i+1;
            end
            maxbs = n - i + 1;
            smallerBlock = [];
            while (1)
                largerBlock = f(A(i:i+bs-1, i:i+bs-1));
                if (sufficientlyBanded(largerBlock, tol))
                    bandwidthBlocks = [bandwidthBlocks, bs/2];
                    break;
                elseif (bs == maxbs)
                    bandwidthBlocks = [bandwidthBlocks, bs];
                    break;
                else
                    smallerBlock = largerBlock;
                    bs = min(bs * 2, maxbs);
                    if (bs > (n-i+1)/2)
                        bs = n-i+1;
                    end
                end
            end
            dimDiagBlocks = [dimDiagBlocks, bs];
            fDiagBlocks{cont} = sparse(largerBlock);

            if (size(smallerBlock, 1) ~= bs/2)
                smallerBlock = f(A(i:i+floor(bs/2)-1, i:i+floor(bs/2)-1));
            end
            fSmallerBlocks{1, cont} = smallerBlock;
            fSmallerBlocks{2, cont} = f(A(i+floor(bs/2):i+bs-1, i+floor(bs/2):i+bs-1));

            i = i + bs;
            if (i < n)
                splittingPoints = [splittingPoints, i];
            end   
            
            cont = cont+1;
        end
        
        if (debug == 1)
            disp(dimDiagBlocks)
        end

        % Compute updates
        Updates = {};
        firstIndexUpdate = [];
        dimUpdates = [];
        allUpdatesOK = 1;
        for j = 1:length(splittingPoints)
            i = splittingPoints(j);
            bs1 = floor(dimDiagBlocks(j)/2);
            bs2 = floor(dimDiagBlocks(j+1)/2);
            
            update = f(A(i-bs1 : i+bs2-1, i-bs1 : i+bs2-1)) - blkdiag(fSmallerBlocks{2, j}, fSmallerBlocks{1, j+1});
            
            if (updateConverged(update, bs1, bs2, tol, lag) == 0)
                allUpdatesOK = 0;
                nsteps = 0;
                while (updateConverged(update, bs1, bs2, tol, lag) == 0 && (bs1 < i-1 || bs2 < n-i+1))
                    bs1 = min(bs1 + lag, i-1);
                    bs2 = min(bs2 + lag, n-i+1);
                    update = f(A(i-bs1 : i+bs2-1, i-bs1 : i+bs2-1)) - ...
                        blkdiag(f(A(i-bs1 : i-1, i-bs1 : i-1)), f(A(i : i+bs2-1, i : i+bs2-1)));
                    if (debug > 0)
                        nsteps = nsteps + 1;
                    end
                end
                if (debug > 0)
                    fprintf("An update took %d extra steps to converge\n", nsteps)
                end
            end
            Updates{j} = update;
            firstIndexUpdate = [firstIndexUpdate, i-bs1];
            dimUpdates = [dimUpdates, bs1 + bs2];
        end
        
        % Assemble matrix
        F = blkdiag(fDiagBlocks{:});
        if (isempty(firstIndexUpdate))
            return;
        end
        if (allUpdatesOK)
            % Quicker solution
            i = firstIndexUpdate(1);
            U = blkdiag(sparse(i-1, i-1), Updates{:});
            U = blkdiag(U, sparse(n-size(U, 1), n-size(U, 1)));
            F = F + U;            
        else
            for j = 1:length(firstIndexUpdate)
                i = firstIndexUpdate(j);
                dim = dimUpdates(j);
                F(i:i+dim-1, i:i+dim-1) = F(i:i+dim-1, i:i+dim-1) + Updates{j};
            end
        end
    end
    % F = F .* (F > tol) + F .* (F < -tol);
end

% Check if an update has converged
function x = updateConverged(B, p, q, tol, lag)
    if (max(max(abs(B([1:lag, p+q-lag+1 : p+q], :))))>tol || max(max(abs(B(:,[1:lag, p+q-lag+1 : p+q]))))>tol)
        x = 0;
    else
        x = 1;
    end
end

% Check if a matrix is "approximately banded"
function x = sufficientlyBanded(B, tol)
    n = size(B, 1);
    if (max(max(abs(tril(B(floor(n/2)+1:n,1:floor(n/2)))))) > tol || max(max(abs(triu(B(1:floor(n/2),floor(n/2)+1:n))))) > tol)
        x = 0;
    else
        x = 1;
    end
end