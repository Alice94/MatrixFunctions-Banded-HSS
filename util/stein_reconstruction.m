function T = stein_reconstruction(G,B)
% T = stein_reconstruction(G,B)
%
% Reconstruct the Toeplitz matrix T from a generator.

n = size(G,1);
T = G*B';

for j = 1:(n-1)
    T((j+1):end, j+1) = T((j+1):end, j+1) + T(j:(end-1), j);
end

T = T';
for j = 1:(n-1)
    T((j+2):end, j+1) = T((j+2):end, j+1) + T((j+1):(end-1), j);
end

T = T';

end