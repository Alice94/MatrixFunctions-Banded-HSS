function [T, Q] = TLanczos(A, n_it, u)

n = size(A, 1);
if (n_it > n)
    n_it = n;
end

x = u/norm(u);
Q = [];
xold = sparse(n,1);
beta = 0;

for j=1:n_it
    alpha = x'*A*x;
    Q = [Q, x];
    r = A*x - alpha*x - beta*xold;
    % Reorthogonalization:
    r = r - Q*(Q'*r);

    % Gauss quadrature rule
    if (j == 1)
        T = alpha;
    else
        T(j,j) = alpha;
        T(j-1,j) = beta;
        T(j,j-1) = beta;
    end

    % Update stuff
    beta = norm(r);
    xold = x;

    x = r/beta;
end