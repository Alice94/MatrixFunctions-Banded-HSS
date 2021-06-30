function [Gs, Bs] = toeplksquare(G, B, alg)
% [Gs, Bs] = toeplksquare(G, B, alg)
%
% Compute generator for squared TL matrix.
%
% Possible choices for paramter 'alg':
%   'full' -- traditional matrix vector product with reconstructed matrix
%   'fft'  -- FFT based matrix vector product

if nargin < 3 || isempty(alg)
    alg = 'full';
end

switch alg
    case 'full'
        [Gs, Bs] = tlsquare_full(G,B);
    case 'fft'
        [Gs, Bs] = tlsquare_fft(G,B);
    otherwise
        error('Invalid choice for alg parameter');
end

end

function [Gs, Bs] = tlsquare_fft(G,B)



n = size(G,1);

onesvec = ones(n,1);

G2_part1 = vapply(G, 'inv');
G2_part1 = toeplkmult(G, B, G2_part1);
G2_part1 = vapply(G2_part1);

G2_part2 = toeplkmult(G, B, -onesvec);
G2_part2 = vapply(G2_part2);

Gs = [G2_part1, G, -G2_part2];

B2_part1 = vapply(B, 'inv');
B2_part1 = toeplkmult(B, G, B2_part1);
B2_part1 = vapply(B2_part1);

B2_part2 = toeplkmult(B, G, -onesvec);
B2_part2 = vapply(B2_part2);

Bs = [B, B2_part1, B2_part2];


end

function [Gs, Bs] = tlsquare_full(G, B)

T = stein_reconstruction(G,B);

G2_part1 = vapply(G, 'inv');
G2_part1 = T*G2_part1;
G2_part1 = vapply(G2_part1);

G2_part2 = - sum(T,2);
G2_part2 = vapply(G2_part2);

Gs = [G2_part1, G, -G2_part2];

B2_part1 = vapply(B, 'inv');
B2_part1 = T' * B2_part1;

B2_part1 = vapply(B2_part1);
B2_part2 = - sum(T,1)';
B2_part2 = vapply(B2_part2);

Bs = [B, B2_part1, B2_part2];

end
