function y = ftimes(x,s,eta)
%FTIMES Fourier matrix product.
%   FTIMES multiplies the unitary Fourier matrix F times a matrix X,
%   having at least two rows.
%
%   Y = FTIMES(X)         computes F*X;
%   Y = FTIMES(X,[])      as above;
%   Y = FTIMES(X,'N')     as above;
%   Y = FTIMES(X,'A')     computes F'*X (applies the adjoint of F).
%
%   If the first parameter is a scalar, it must be a positive integer
%   and the output is a matrix, according to the second argument.
%  
%   Y = FTIMES(n)         returns F which is n-by-n;
%   Y = FTIMES(n,[])      as above;
%   Y = FTIMES(n,'N')     as above;
%   Y = FTIMES(n,'A')     returns  F', i.e., the adjoint of F.
%
%   Y = FTIMES(X,S,ETA) and Y = FTIMES(n,S,ETA) use the unitary matrix
%   diag(w)*F in place of F. The argument S can be 'N', 'A', or [],
%   ETA is a complex unitary scalar, and the phase matrix diag(w) is
%   defined by
%
%      w(j) = ETA^(-(j-1)/n),     j=1,..,n.
%
%   In Matlab notation F is shortly defined as F = fft(eye(n)/sqrt(n).
%
%   See also ctimes, stimes, fft.

% IMPLEMENTATION DETAILS: the test on the first input argument is
%     size(x,1)>1
% so it is NOT possible:
%     1) build F (or W*F, or F' or F'*W') with size 1-by-1
%     2) apply F (1-by-1) to a row vector

%   Antonio Arico' & Giuseppe Rodriguez, University of Cagliari, Italy
%   Email: {arico,rodriguez}@unica.it
%
%   Last revised Dec 11, 2009

%==========================================================================
% parse input
%==========================================================================
% flag1: really assemble the F [or W*F] matrix?
if size(x,1)>1
    n     = size(x,1);
    flag1 = 0;          % NO, I will apply [I]FFT of length n instead :)
elseif ~isscalar(x) || x~=ceil(x) || x<2
    error('ftimes:wrongX','x must be either n-by-m (with n>1), or an integer >1.');
else
    flag1 = 1;          % YES, I will assemble a matrix :(
    n     = x;
end
%--------------------------------------------------------------------------
% Adjoint or Not?
if nargin<2 || isempty(s), s='N'; end
if s~='N' && s~='A'
    error('ftimes:wrongS','S must be ''A''djoint or ''N''ot');
end
%--------------------------------------------------------------------------
% flag2: use W*F?
if nargin<3, 
    flag2 = 0;                         % NO, just use F
elseif abs(abs(eta)-1)>eps
    error('ftimes:wrongETA','|ETA| is not 1.');
else
    flag2 = (mod(angle(eta),2*pi)~=0); % YES, unless eta=1     % remove ~=0
end
%==========================================================================
% compute
%==========================================================================
%if flag2, dW = exp(complex(0,-angle(eta)*(0:n-1)/n)).'; end;
if flag2, dW = exp(-1i*angle(eta)*(0:n-1)/n).'; end;
%
if flag1
    %                                                       ASSEMBLE matrix
    if s=='N'
        if flag2
            %y = spdiags(dW/sqrt(n),0,n,n) * fft(eye(n));%              W*F
            y = ifft(diag(dW'*sqrt(n)))';%                  W*F=(F'*d(w'))' 
        else
            %y = fft(eye(n))/sqrt(n);%                                  F*I
            y = fft(diag(repmat(1/sqrt(n),[n 1])));%                    F*I
        end
    else
        % S='A'
        if flag2
            y = ifft(diag(sqrt(n)*dW'));%                             F'*W'
        else
            %y = sqrt(n)*ifft(eye(n));%                                F'*I
            y = ifft(diag(repmat(sqrt(n),[n 1])));%                    F'*I
        end        
    end
else
    %                                                          APPLY [i]fft
    %x=vector|matrix
    x = full(x);
    if s=='N'
        if flag2
            y = spdiags(dW/sqrt(n),0,n,n) * fft(x);%                W*(F*x)
        else
            y = fft(x)/sqrt(n);%                                        F*x
        end
    else
       % S='A'
       if flag2
           y = ifft(full(spdiags(sqrt(n)*conj(dW),0,n,n)*x));%    F'*(W'*x)
       else
           y = sqrt(n)*ifft(x);%                                       F'*x
       end
    end
end
