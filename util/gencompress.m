function [Gout, Bout, res_sv] = gencompress(G, B, target_r)
% [Gout, Bout, res_sv] = gencompress(G, B, target_r)
% 
% Compress a given generator to smaller rank.


[n,r] = size(G);
assert(all([n,r] == size(B)));


detect = false;
if nargin < 3 || isempty(target_r)
    detect = true;
elseif r <= target_r
    Gout = G;
    Bout = B;
    res_sv = 0.0;
    return;
end

[QG, RG] = qr(G,0);
[QB, RB] = qr(B,0);

[U, S, V] = svd(RG*RB');
s = diag(S);

if detect
    target_r = rank(S);
end

if length(s) <= target_r
    res_sv = 0.0;
else
    res_sv = s(target_r + 1);
end

if length(s) < target_r
    target_r = length(s);
end

% TODO
% Two ways to set up the generator:  split singular values and scale both G
% and B, or keep G truly orthogonal and scale B only.  Should not make a
% huge difference, but could be investigated.

%Gout = QG * U(:,1:target_r) * diag(sqrt(s(1:target_r)));
%Bout = QB * V(:,1:target_r) * diag(sqrt(s(1:target_r)));

Gout = QG * U(:,1:target_r);
Bout = QB * V(:,1:target_r) * diag(s(1:target_r));

end