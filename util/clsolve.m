function [B rcondu p q] = clsolve(G,H,t,s,B,piv)
%CLSOLVE Solution of a Cauchy-like linear system.
%   x = CLSOLVE(G,H,d1,d2,b,piv) computes the solution to the linear
%   system C*x=b, where (G,H) are the generators of the Cauchy-like
%   matrix C and d1, d2 the diagonals of its displacement matrices,
%   i.e.,
%
%        diag(d1) * C - C * diag(d2) = G*H'
%
%   b is the RHS which may contain multiple columns.
%   The parameter piv selects the pivoting technique:
%       0   no pivoting
%       1   partial pivoting (default)
%       2   multiple entries in d2 + partial pivoting
%       3   Sweet and Brent's pivoting
%       4   Gu's pivoting
%       5   complete pivoting
%
%   [x,rcondu] = CLSOLVE(...) also returns an estimate for rcond(U),
%   which is a rough estimate for rcond(C).
%
%   The vectors d1 and d2 must not have entries in common.
%   The vector d2 must not have multiple entries, unless piv=2.
%
%   See also tsolve, tlsolve, thsolve, thlsolve, vsolve, vlsolve.

%   Antonio Arico' & Giuseppe Rodriguez, University of Cagliari, Italy
%   Email: {arico,rodriguez}@unica.it
%
%   Last revised March 25, 2010

% CLSOLVE computes the Schur complement of the matrix A=[C B; -I, 0], that
% is 0+I/C*B= C\B. This is done via the GKO algorithm

%  INPUT  size     description
%
%      G  n-by-dr  left-generator of C
%      H  n-by-dr  right-generator of C
%      t  n-by-1   left-displacement of C
%      s  n-by-1   right-displacement of C
%      B  n-by-kB  right hand side
%    piv  1-by-1   kind of pivoting (0,1,2...)
%
% OUTPUT  size
%
%      x  (n-by-kB) solution
%  rcondu 1-by-1   estimate of 1/cond(U,1) where C=L*U;
%      p  n-by-1   row pivoting vector
%      kB n-by-1   col pivoting vector

warning('drsolve:clsolve:missingMEX', ...
    'MEX file missing: computation may be slow.')

%
% check input
if (nargin<6) || isempty(piv), piv = 1; end % partial pivoting is the default
if (nargin<5)
    error('drsolve:clsolve:tooFewArguments', 'Too few input arguments.')
end

%
% decode additional parameters
addpar = floor(piv/10);
piv    = mod(piv,10);

%
% refine input
t = t(:); % now is col vector
s = s(:); % "

%
% check size of input
N  = [ size(t,1) size(s,1) size(G,1) size(H,1) size(B,1) ];
DR = [ size(G,2) size(H,2) ];
if ( min(N)<max(N) || min(DR)<max(DR) )
    error('drsolve:clsolve:inconsistentDimensions', ...
        'Input arrays have inconsistent dimensions.')
end

%
% get dimensions
n  = N(1);		% size of linear system
dr = DR(1);		% displacement rank
if n < dr
    error('drsolve:clsolve:badGenerators', ...
        'Generators have too many columns.')
end

%
% check if the mosaic matrix is reconstructable
%   (1,1) => intersect(t,s) = emptyset
if numel(intersect(t,s)) > 0
    error('drsolve:clsolve:notReconstructable', ...
        'Partially reconstructable matrix.')
end
%   (2,1) => no repetitions in s [note that diag(block_(2,1))==-ones(n,1) ]
if (numel(unique(s)) < n) && (piv ~= 2)
    warning('drsolve:clsolve:multipleS', ...
        'Vector d2 has multiple entries, try piv=2.')
end

%
% generators of [ C B; -I 0 ]
% --
% large_G  = [ G spdiags(tC,0,n,n)*B; zeros(n,dr+kB) ];
% --
% large_H  = [ H' zeros(dr,kB); zeros(kB,n) eye(kB) ];
H = H';			% now diag(t)*C-C*diag(s) = G*H;

%
% pivoting permutations
p = (1:n).';		% rows permutation
q = (1:n).';		% cols permutation

%
% rcond estimate locations
uinvnm  = 0;
ucolsum = zeros(1,n);

l  = zeros(n,1); 
u1 = zeros(1,n); % use: align on the left

switch piv


    case 0	% no pivoting
        for k = 1:n
            %
            l = (G*H(:,k)) ./ ([s(1:k-1);t(k:n)]-s(k)); % #=n
            %
            pivot = l(k);
            l(k) = -1;
	    %
            if abs(pivot) == 0
                warning('drsolve:clsolve:nullDiagonalElement', ...
                        'Null diagonal element. Try to activate pivoting.')
            end
            %
            u1(1:n-k) = (G(k,:)*H(:,k+1:n)) ./ (t(k)-s(k+1:n).'); % #=n-k
            %
            g1_pivot = G(k,:)/pivot;
            G(k,:)   = 0;
            G        = G-l*g1_pivot;
            %
            g2_pivot = B(k,:)/pivot;
            B(k,:)   = 0;
            B        = B-l*g2_pivot;
            %
            h1_pivot   = H(:,k)/pivot;
            H(:,k+1:n) = H(:,k+1:n) - h1_pivot*u1(1:n-k);
            %
            uinvnm = max([uinvnm; ...
                sum(abs(real(l(1:k)))+abs(imag(l(1:k)))) / abs(pivot) ]);%~
            ucolsum(k:n) = ucolsum(k:n) + ...
                [ abs(real(pivot))+abs(imag(pivot)), ...
                  abs(real(u1(1:n-k)))+abs(imag(u1(1:n-k))) ];
            %
        end


    case 1	% partial pivoting
        for k = 1:n
            %
            l = (G*H(:,k)) ./ ([s(1:k-1);t(k:n)]-s(k)); % #=n
            % search for pivot
            [lmax imax] = max( abs(real(l(k:n))) + abs(imag(l(k:n))) );
            % singular matrix
            if lmax == 0
                warning('drsolve:clsolve:singularMatrix', ...
                    'Matrix is singular to working precision.')
            end
            % pivoting
            if imax > 1
                imax  = imax+(k-1);
                l([k imax])   = l([imax k]);
                t([k imax])   = t([imax k]);
                G([k imax],:) = G([imax k],:);
                B([k imax],:) = B([imax k],:);
                p([k imax])   = p([imax k]);   % useless: just to know p
            end
            %
            pivot = l(k);
            l(k) = -1;
            %
            u1(1:n-k) = (G(k,:)*H(:,k+1:n)) ./ (t(k)-s(k+1:n).'); % #=n-k
            %
            g1_pivot = G(k,:)/pivot;
            G(k,:)   = 0;
            G        = G-l*g1_pivot;
            %
            g2_pivot = B(k,:)/pivot;
            B(k,:)   = 0;
            B        = B-l*g2_pivot;
            %
            h1_pivot   = H(:,k)/pivot;
            H(:,k+1:n) = H(:,k+1:n) - h1_pivot*u1(1:n-k);
            %
            uinvnm = max([uinvnm; ...
                sum(abs(real(l(1:k)))+abs(imag(l(1:k)))) / abs(pivot) ]);%~
            ucolsum(k:n) = ucolsum(k:n) + ...
                [ abs(real(pivot))+abs(imag(pivot)), ...
                  abs(real(u1(1:n-k)))+abs(imag(u1(1:n-k))) ];
            %
        end

        
    case 2	% partial pivoting with multiple s's
        [s, q] = sort(s);
        [ss, a, b] = unique(s,'first');
        [ss, c, d] = unique(s,'last');
        i1 = a(b);
        i2 = c(d);
        delta = max(diff([i1; numel(s)+1]));
        %
        if delta > dr
            error('drsolve:clsolve:singularMatrix', ...
                  'Cauchy-like matrix is singular.')
        end
        H = H(:,q);
        for k = 1:n
            %
            l(k:n)       = (G(k:n,:)      *H(:,k)) ./ (t(k:n)      -s(k)); 
            l(1:i1(k)-1) = (G(1:i1(k)-1,:)*H(:,k)) ./ (s(1:i1(k)-1)-s(k));
            %
            if i1(k) < k
                l(i1(k):k-1) = H(k-i1(k),i1(k):k-1).';
            end
            % -----
            %
            [lmax imax] = max( abs(real(l(k:n))) + abs(imag(l(k:n))) );
            % singular matrix
            if lmax == 0
                warning('drsolve:clsolve:singularMatrix', ...
                        'Matrix is singular to working precision.')
            end
            % pivoting
            if imax > 1
                imax  = imax+(k-1);
                l([k imax])   = l([imax k]);
                t([k imax])   = t([imax k]);
                G([k imax],:) = G([imax k],:);
                B([k imax],:) = B([imax k],:);
                p([k imax])   = p([imax k]);   % useless: just to know p
            end
            %
            pivot = l(k);
            l(k) = -1;
            %
            u1(1:n-k) = (G(k,:)*H(:,k+1:n)) ./ (t(k)-s(k+1:n).'); % #=n-k
            %
            g1_pivot = G(k,:)/pivot;
            G(k,:)   = 0;
            G        = G-l*g1_pivot;
            %
            g2_pivot = B(k,:)/pivot;
            B(k,:)   = 0;
            B        = B-l*g2_pivot;
            %
            h1_pivot   = H(:,k)/pivot;
            H(:,k+1:n) = H(:,k+1:n) - h1_pivot*u1(1:n-k);
            %
            if k < i2(k)
                pos = k-i1(k)+1;
                len = i2(k)-i1(k);
                H(pos:len,k)       = 0;
                H(pos:len,i1(k):k) = H(pos:len,i1(k):k) - ...
                                     u1(1:i2(k)-k).'*(l(i1(k):k).'/pivot);
            end
            %
            uinvnm = max([uinvnm; ...
                sum(abs(real(l(1:k)))+abs(imag(l(1:k)))) / abs(pivot) ]);%~
            ucolsum(k:n) = ucolsum(k:n) + ...
                [ abs(real(pivot))+abs(imag(pivot)), ...
                  abs(real(u1(1:n-k)))+abs(imag(u1(1:n-k))) ];
            %
        end


    case 3	% Sweet&Brent's pivoting
        for k = 1:n
            % search for pivot
            l(k:n)    = ( G(k:n,:)*H(:,k)     )  ./  ( t(k:n)-s(k)       ); % #=n-k+1
            u1(1:n-k) = ( G(k,:)  *H(:,k+1:n) )  ./  ( t(k)  -s(k+1:n).' ); % #=n-k
            %
            [p1,i1] = max( abs(real(l(k:n))) + abs(imag(l(k:n))) );
            [p2,i2] = max( abs(real(u1(1:n-k))) + abs(imag(u1(1:n-k))) );
            r_ind = i1+k-1;
            c_ind = i2+k;
            %
            if ( max(p1,p2)==0 )
                 warning('drsolve:clsolve:singularMatrix', ...
                         'Matrix is singular to working precision.')
            elseif ( p2>p1 )
                % do column pivoting
                s(  [c_ind,k]) = s(  [k,c_ind]);
                H(:,[c_ind,k]) = H(:,[k,c_ind]);
                q(  [c_ind,k]) = q(  [k,c_ind]);
                %
		u1(i2) = l(k);                                 % update
                l(k:n) = ( G(k:n,:)*H(:,k) ) ./ (t(k:n)-s(k)); % recompute
            else
                % do row pivoting
                if r_ind ~= k
                    l([r_ind,k]  ) = l([k,r_ind]  );
                    t([r_ind,k]  ) = t([k,r_ind]  );
                    G([r_ind,k],:) = G([k,r_ind],:);
                    B([r_ind,k],:) = B([k,r_ind],:);
                    p([r_ind,k]  ) = p([k,r_ind]  );
                    u1(1:n-k) = (G(k,:)*H(:,k+1:n)) ./ (t(k)-s(k+1:n).'); % recompute
                end
            end
	    %
            l(1:k-1)  = (G(1:k-1,:)*H(:,k)) ./ (s(1:k-1)-s(k) ); 
            %
            pivot = l(k);
            l(k) = -1;
            %
            g1_pivot = G(k,:)/pivot;
            G(k,:)   = 0;
            G        = G-l*g1_pivot;
            %
            g2_pivot = B(k,:)/pivot;
            B(k,:)   = 0;
            B        = B-l*g2_pivot;
            %
            h1_pivot   = H(:,k)/pivot;
            H(:,k+1:n) = H(:,k+1:n) - h1_pivot*u1(1:n-k);
            %
            uinvnm = max([uinvnm; ...
                sum(abs(real(l(1:k)))+abs(imag(l(1:k)))) / abs(pivot) ]);%~
            ucolsum(k:n) = ucolsum(k:n) + ...
                [ abs(real(pivot))+abs(imag(pivot)), ...
                  abs(real(u1(1:n-k)))+abs(imag(u1(1:n-k))) ];
            %
        end


    case 4	% GU
        if addpar
            Jmax_step = addpar;
        else
            Jmax_step = 10;		% default value
        end
        for k = 1:n
            % column pivoting
            if (mod(k,Jmax_step) == 1) && (k <= n-dr+1)
                [ G(k:n,:) R ] = qr(G(k:n,:),0);
		G(1:k-1,:) = G(1:k-1,:) / R;
                H(:,k:n) = R*H(:,k:n);
                [ nmax c_ind ] = max(sum( abs(real(H(:,k:end)))...
                                        + abs(imag(H(:,k:end))), 1));
                c_ind = c_ind+(k-1);
                if c_ind ~= k
                    s(  [c_ind,k]) = s(  [k,c_ind]);
                    H(:,[c_ind,k]) = H(:,[k,c_ind]);
                    q(  [c_ind,k]) = q(  [k,c_ind]);
                end
            end
            %
            l = (G*H(:,k)) ./ ([s(1:k-1);t(k:n)]-s(k)); % #=n
            % search for pivot
            [lmax imax] = max( abs(real(l(k:n))) + abs(imag(l(k:n))) );
            % singular matrix
            if lmax == 0
                warning('drsolve:clsolve:singularMatrix', ...
                        'Matrix is singular to working precision.')
            end
            % pivoting
            if imax > 1
                imax  = imax+(k-1);
                l([k imax])   = l([imax k]);
                t([k imax])   = t([imax k]);
                G([k imax],:) = G([imax k],:);
                B([k imax],:) = B([imax k],:);
                p([k imax])   = p([imax k]);   % useless: just to know p
            end
            %
            pivot = l(k);
            l(k) = -1;
            %
            u1(1:n-k) = (G(k,:)*H(:,k+1:n)) ./ (t(k)-s(k+1:n).'); % #=n-k
            %
            g1_pivot = G(k,:)/pivot;
            G(k,:)   = 0;
            G        = G-l*g1_pivot;
            %
            g2_pivot = B(k,:)/pivot;
            B(k,:)   = 0;
            B        = B-l*g2_pivot;
            %
            h1_pivot   = H(:,k)/pivot;
            H(:,k+1:n) = H(:,k+1:n) - h1_pivot*u1(1:n-k);
            %
            uinvnm = max([uinvnm; ...
                sum(abs(real(l(1:k)))+abs(imag(l(1:k)))) / abs(pivot) ]);%~
            ucolsum(k:n) = ucolsum(k:n) + ...
                [ abs(real(pivot))+abs(imag(pivot)), ...
                  abs(real(u1(1:n-k)))+abs(imag(u1(1:n-k))) ];
            %
        end


    case 5	% complete pivoting
        max_c  = zeros(1,n); % align at right
        rv_ind = zeros(1,n); % align at right
        Mj     = zeros(n,1); % align at bottom
        for k = 1:n
            %
            for j=k:n
               Mj(k:n) = (G(k:n,:)*H(:,j)) ./ (t(k:n)-s(j));
               [ max_c(j), rv_ind(j) ] = max( abs(real(Mj(k:n))) + ...
                                              abs(imag(Mj(k:n))) );
            end
            [apivot,c_ind] = max(max_c(k:n));
            c_ind = c_ind         + (k-1);    % is in [k:n]
            r_ind = rv_ind(c_ind) + (k-1);    % is in [k:n]
            % singular matrix
            if apivot == 0
                warning('drsolve:clsolve:singularMatrix', ...
                        'Matrix is singular to working precision.')
            end
            % pivoting
            if r_ind ~= k
                t([r_ind,k]  ) = t([k,r_ind]  );
                G([r_ind,k],:) = G([k,r_ind],:);
                B([r_ind,k],:) = B([k,r_ind],:);
                p([r_ind,k]  ) = p([k,r_ind]  ); % useless: just to know p
            end
            if c_ind ~= k
                s(  [c_ind,k]) = s(  [k,c_ind]);
                H(:,[c_ind,k]) = H(:,[k,c_ind]);
                q(  [c_ind,k]) = q(  [k,c_ind]);
            end
            %
            l = (G*H(:,k)) ./ ([s(1:k-1);t(k:n)]-s(k)); % #=n
            %
            pivot = l(k);
            l(k) = -1;
            %
            u1(1:n-k) = (G(k,:)*H(:,k+1:n)) ./ (t(k)-s(k+1:n).'); % #=n-k
            %
            g1_pivot = G(k,:)/pivot;
            G(k,:)   = 0;
            G        = G-l*g1_pivot;
            %
            g2_pivot = B(k,:)/pivot;
            B(k,:)   = 0;
            B        = B-l*g2_pivot;
            %
            h1_pivot   = H(:,k)/pivot;
            H(:,k+1:n) = H(:,k+1:n) - h1_pivot*u1(1:n-k);
            %
            uinvnm = max([uinvnm; ...
                sum(abs(real(l(1:k)))+abs(imag(l(1:k)))) / abs(pivot) ]);%~
            ucolsum(k:n) = ucolsum(k:n) + ...
                [ abs(real(pivot))+abs(imag(pivot)), ...
                  abs(real(u1(1:n-k)))+abs(imag(u1(1:n-k))) ];
            %
        end


    otherwise
        error('drsolve:clsolve:pivotingNotSupported', ...
            'This kind of pivoting is unsupported.')

        
end

if piv >= 2
        B(q,:) = B;
end

unorm = max(ucolsum);
rcondu = 1/unorm/uinvnm;

if rcondu < eps
    warning('drsolve:clsolve:badConditioning', ...
        ['Matrix is close to singular or badly scaled.\n', ...
        sprintf('         Results may be inaccurate. RCONDU = %g.',rcondu)])
end

