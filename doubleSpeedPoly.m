addpath('util')
addpath('util/matrices')

rng(1)
fontSize = 13;

% symmetric matrix
n = 1024;
% A = spdiags(ones(n, 1) * [1 -2 1], -1:1, n, n);
a1 = randn(n, 1);
a2 = randn(n, 1);
A = spdiags([a1, a2, a1], -1:1, n, n);
A = A / normest(A, 1e-2);

f = @expm;
fA = f(A);

errTrace = [];
errFull = [];
errDiag = [];
ms = 2:12;

for m = ms
    s = 2*m;
    approxfA = Splitting(A,1,f,s,1e30,0);
    
    errDiag = [errDiag, norm(diag(approxfA) - diag(fA))];
    errTrace = [errTrace, abs(trace(fA) - trace(approxfA))];
    errFull = [errFull, norm(fA - approxfA, 'fro')];    
end

figure('Position', [0 0 500 400])

semilogy(ms, errFull, 'linewidth', 2)
hold on
semilogy(ms, errDiag, 'linewidth', 2)
semilogy(ms, errTrace, 'linewidth', 2)

legend('Full f(A)', 'Diagonal', ...
    'Trace', 'Location', 'best')
xlabel('m')
ylabel('error')
set(gca,'Fontsize',fontSize);
% saveas(gcf, 'doubleSpeedBanded1', 'epsc')
hold off

% Random banded matrix (pentadiagonal, non symmetric)
n = 1024;
r = 4;
A = spdiags(randn(n, 5), -2:2, n, n);
A = A / normest(A, 1e-2);

f = @expm;
fA = f(A);

errTrace = [];
errFull = [];
errDiag = [];
ms = 2:12;

for m = ms
    s = 4*m;
    approxfA = Splitting(A,2,f,s,1e30,0);
    
    errDiag = [errDiag, norm(diag(approxfA) - diag(fA))];
    errTrace = [errTrace, abs(trace(fA) - trace(approxfA))];
    errFull = [errFull, norm(fA - approxfA, 'fro')];    
end

figure('Position', [0 0 500 400])

semilogy(ms, errFull, 'linewidth', 2)
hold on
semilogy(ms, errDiag, 'linewidth', 2)
semilogy(ms, errTrace, 'linewidth', 2)

legend('Full f(A)', 'Diagonal', ...
    'Trace', 'Location', 'best')
xlabel('m')
ylabel('error')
set(gca,'Fontsize',fontSize);
hold off
% saveas(gcf, 'doubleSpeedBanded2', 'epsc')



%%
fontSize = 16;


% Random symmetric matrix
n = 1024;
r = 1;
% [Q, ~] = qr(randn(n));
% D = diag(rand(n, 1));
% A = Q*D*Q';
% [Q, ~] = qr(randn(r));
% D = diag(rand(r, 1));
A = randn(n);
A = A + A';
A = A/norm(A);
B = randn(n, r);
B = B / norm(B);
J = 1;

trueUpdate = expm(-A-B*J*B') - expm(-A);

errTrace = [];
errUpdate = [];
errDiag = [];
ms = 1:14;

for m = ms
    % Construct (polynomial) block Krylov subspace of dimension m
    [Q, ~] = qr(B, 0);
    for i=1:m-1
        W = A * Q(:, 1+(i-1)*r : i*r);
        W = W - Q * Q' * W;
        [W, ~] = qr(W, 0);
        Q = [Q, W];        
    end
    G = Q'*A*Q;
    approxUpdate = expm(-G-(Q'*B)*J*(B'*Q)) - expm(-G);
    
    errDiag = [errDiag, norm(diag(Q*approxUpdate*Q') - diag(trueUpdate))];
    errTrace = [errTrace, abs(trace(trueUpdate) - trace(approxUpdate))];
    errUpdate = [errUpdate, norm(trueUpdate - Q*approxUpdate*Q', 'fro')];    
end

figure('Position', [0 0 600 600])

semilogy(ms, errUpdate, 'linewidth', 3)
hold on
semilogy(ms, errDiag, 'linewidth', 3)
semilogy(ms, errTrace, 'linewidth', 3)

legend('Full update', 'Diagonal', ...
    'Trace', 'Location', 'best')
xlabel('m')
ylabel('error')
set(gca,'Fontsize',fontSize);
% saveas(gcf, 'doubleSpeedHermitian1', 'epsc')
hold off

%%
% Random nonsymmetric matrix
n = 1024;
[Q, ~] = qr(randn(n));
[V, ~] = qr(randn(n));
D = diag(rand(n, 1));
A = Q*D*V;
B = randn(n, 1);
B = B/norm(B);
C = B;

trueUpdate = expm(-A-B*C') - expm(-A);

errTrace = [];
errUpdate = [];
errDiag = [];
ms = 1:14;

for m = ms
    % Construct (polynomial) block Krylov subspace of dimension m
    [Q1, ~] = qr(B, 0);
    for i=1:m-1
        W = A * Q1(:, 1+(i-1) : i);
        W = W - Q1 * Q1' * W;
        [W, ~] = qr(W, 0);
        Q1 = [Q1, W];        
    end
    Q2 = Q1;
    [Q2, ~] = qr(C, 0);
    for i=1:m-1
        W = A' * Q2(:, 1+(i-1) : i);
        W = W - Q2 * Q2' * W;
        [W, ~] = qr(W, 0);
        Q2 = [Q2, W];        
    end
    G = Q1'*A*Q1;
    H = Q2'*A*Q2;
    QQ = blkdiag(Q1, Q2);
    AA = QQ' * [A, B*C'; zeros(n), A + B*C'] * QQ;
    FF = expm(-AA);
    approxUpdate = FF(1:size(AA, 1)/2, (size(AA, 1)/2+1):end);
    
    errDiag = [errDiag, norm(diag(Q1*approxUpdate*Q2') - diag(trueUpdate))];
    errTrace = [errTrace, abs(trace(trueUpdate) - trace(Q1*approxUpdate*Q2'))];
    errUpdate = [errUpdate, norm(trueUpdate - Q1*approxUpdate*Q2', 'fro')];    
end

figure('Position', [0 0 600 600])

semilogy(ms, errUpdate, 'linewidth', 3)
hold on
semilogy(ms, errDiag, 'linewidth', 3)
semilogy(ms, errTrace, 'linewidth', 3)

legend('Full update', 'Diagonal', ...
    'Trace', 'Location', 'best')
xlabel('m')
ylabel('error')
set(gca,'Fontsize',fontSize);
% saveas(gcf, 'doubleSpeedHermitian2', 'epsc')
hold off

%%
% This does not work for rational Krylov subspaces

% 1D Laplacian
n = 1024;
r = 2;
A = spdiags(ones(n, 1) * [-1, 2, -1], -1:1, n, n);
B = sparse(n, 2);
B(n/2, 1) = 1;
B(n/2+1, 2) = 1;
J = [0, 1; 1, 0];
A = A;
coso = max(eig(A))/min(eig(A));

f = @(X) sqrtm(full(X));
trueUpdate = f(A + B*J*B') - f(A);

errTrace = [];
errUpdate = [];
errDiag = [];
ms = 1:30;

for m = ms
    % Construct block Krylov subspace of dimension m
    obj = rkfun.gallery('sqrt', m, coso);
    xi = poles(obj)' * min(eig(A));
%     xi = [inf * ones(1, floor((m-1)/2)), zeros(1, ceil((m-1)/2))];
%     xi = -1 * ones(1, m-1);
    [Q, ~, ~, ~] = rk_krylov(A, B, xi);
    G = Q'*A*Q;
    approxUpdate = f(G+(Q'*B)*J*(B'*Q)) - f(G);
    
    errDiag = [errDiag, norm(diag(Q*approxUpdate*Q') - diag(trueUpdate))];
    errTrace = [errTrace, abs(trace(trueUpdate) - trace(approxUpdate))];
    errUpdate = [errUpdate, norm(trueUpdate - Q*approxUpdate*Q', 'fro')];    
end

figure('Position', [0 0 600 600])
semilogy(ms, errUpdate, 'linewidth', 3)
hold on
semilogy(ms, errDiag, 'linewidth', 3)
semilogy(ms, errTrace, 'linewidth', 3)

legend('Full update', 'Diagonal', ...
    'Trace', 'Location', 'best')
xlabel('m')
ylabel('error')
set(gca,'Fontsize',fontSize);
hold off
% saveas(gcf, 'doubleSpeedHermitian3', 'epsc')

% close all










