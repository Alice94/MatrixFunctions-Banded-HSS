function [V, H, params, lucky] = poly_krylov(varargin)
%EK_KRYLOV Extended Krylov projection of a matrix A.
%
% [V, K, H, params] = POLY_KRYLOV(A, B) construct the extended Krylov
%     subspace spanned by [B, A*B, ...]. The matrix V is an orthogonal
%     basis for this space, and K and H are block upper Hessenberg
%     rectangular matrices satisfying
%
%        A * V * K = V * H                                          (1)
%
% [V, K, H, params] = POLY_KRYLOV(V, K, H, PARAMS) enlarges a
%     Krylov subspace generated with a previous call to POLY_KRYLOV by adding
%     infinity pole pair. The resulting space will satisfy
%     the same relation (1).
% We assume that B has orthonormal columns
%
% Note: lucky is a flag parameter that highlight the lucky breakdown of lanczos
%
if nargin ~= 2 && nargin ~= 3
    error('Called with the wrong number of arguments');
end

if nargin == 2
    % Start to construct the extended Krylov space
    [V, H, params, lucky] = poly_krylov_start(varargin{:});
else
    % Enlarge the space that was previously built
    [V, H, params, lucky] = poly_krylov_extend(varargin{:});
end

end

function [V, H, params, lucky] = poly_krylov_start(A, b)

if ~isstruct(A)
    m = size(A, 1);
    n = size(A, 2);
    
    if m ~= n
        error('The matrix A should be square');
    end
    
    if n ~= size(b, 1)
        error('The block vector b has wrong number of rows');
    end
end

bs = size(b, 2);

% Construct a basis for the column span of b
[V, ~] = qr(b, 0);
% V = b;

H = zeros(bs, 0);

[V, H, w, lucky] = add_inf_pole (V, H, A, V);

% Save parameters for the next call
params = struct();
params.last = w;
params.A = A;
end

function [V, H, params, lucky] = poly_krylov_extend(V, H, params)
    w  = params.last;
    A  = params.A;

    if (size(w, 2) > 0)
        [V, H, w, lucky] = add_inf_pole (V, H, A, w);
    else
        lucky = 1;
    end

    params.last = w;
end

%
% Utility routine that adds an infinity pole to the space. The vector w is
% the continuation vector.
%
function [V, H, w, lucky] = add_inf_pole(V, H, A, w)
	lucky_tol = 1e-14; % tolerance for detecting lucky breakdowns
	lucky = false;
	bs = size(w, 2);

	if isstruct(A)
		w = A.multiply(1.0, 0.0, w);
	else
		w = A * w;
	end

	% Perform orthogonalization with modified Gram-Schimidt
	[w, h] = mgs_orthogonalize(V, w);
    
    [U, C, p] = qr(full(w), 0);
    bsnew = sum(abs(diag(C)) > lucky_tol);
    bsnew = full(bsnew);
%     if (bsnew < bs)
%         fprintf("Deflation needed, old block size = %d, new block size = %d\n", bs, bsnew)
%     end
    
    P(p, :) = speye(bs);
    c = C(1:bsnew, :) * P';
    w = U(:,1:bsnew);
    
	% Enlarge H
	H(size(H, 1) + bsnew, size(H, 2) + bs) = 0;

	H(1:end-bsnew, end-bs+1:end) = h;

	if bsnew == 0
		lucky = true;
    end
    
    [w, hh] = mgs_orthogonalize(V, w);
    H(1:end-bsnew, end-bs+1:end) = H(1:end-bsnew, end-bs+1:end) + hh*c;
    
    [w, cc] = qr(full(w), 0);
	H(end-bsnew+1:end, end-bs+1:end) = cc*c;

	V = [V, w];
%     disp(norm(w - V * (V' * w), 'fro'))
end

%
% Modified Gram-Schmidt orthogonalization procedure.
%
% Suggested improvements: work with block-size matrix vector products to
% get BLAS3 speeds.
%
function [w, h] = mgs_orthogonalize(V, w)
    h = V' * w;
    w = w - V * h;
%     h1 = V' * w;
%     h = h + h1;
%     w = w - V * h1;
end

