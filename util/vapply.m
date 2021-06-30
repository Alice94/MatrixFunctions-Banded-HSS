function v = vapply(x, flag)
% v = vapply(x)
%
% Compute the application of V := Z - I to x, i.e. compute V*x.  Here Z is
% the downshift matrix, and I the identiy.
%
% v = vapply(x, 'inv')
%
% Computes v = inv(V) * v

inverse = false;

[n,b] = size(x);

if nargin >= 2
    if ~strcmp(flag, 'inv')
        error('expmt:InconsistentInput', 'Unknown input parameter');
    end
    inverse = true;
end

if inverse
    v = -cumsum(x,1);
else
    v = [zeros(1,b); x(1:end-1,:)] - x;
end

end