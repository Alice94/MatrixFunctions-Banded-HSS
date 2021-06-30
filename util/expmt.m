function [G, B, drank] = expmt(c, r, varargin)
% [G, B] = expmt(c, r)
%
% Matrix exponential of a Toeplitz matrix.

p = inputParser;

% Paramter setup

% Which approximation logic? Default: negative real spectrum.
addParameter(p, 'alg', 'sexpm', @(x) ischar(x) && ...
    (strcmp(x, 'sexpm') || strcmp(x, 'matlab') ));
% Full output matrix requested?  Allows also to switch to full in squaring
% phase.
addParameter(p, 'full', false, @(x) islogical(x));
% Verbosity for debug output.
addParameter(p, 'dbglvl', 0, @(x) any(x == [0,1,2,3]));


% Pull some values into workspace
parse(p, varargin{:});
dbglvl = p.Results.dbglvl;
do_full = p.Results.full;

n = length(c);
c = c(:);
r = r(:);

if dbglvl >= 1
    fprintf('Matrix size %d, norm r=%.2e, norm c=%.2e\n', ...
        n, norm(r), norm(c));
end

switch p.Results.alg
    case 'sexpm'
        % Subdiagonal approx a la Guettel/Nakatsukasa
        [G,B, drank] = subdiagonal_pade(c, r, dbglvl);
        
    case 'matlab'
        % Diagonal approx as in Matlab's expm (pre-2015b),
        % based on Highams 2005 paper.
        [G,B, drank] = diagonal_pade_matlab(c, r, dbglvl, do_full);
    otherwise
        % Cannot happen if parameters are checked correctly
        assert(false);
end

% Should the output be the full exponential?
% Convention: if B is empty, then G already holds the full expm(T).
if do_full && ~isempty(B)
    G = stein_reconstruction(G, B);
    B = [];
end

end


function [G, B, drank] = diagonal_pade_matlab(c, r, dbglvl, do_full)
ttotal = tic;

tnorm = toeplitz_1norm(c,r);
% Rat degrees along with theta values, see paper by Higham 2005
m_vals = [3 5 7 9 13];
theta = [
    1.495585217958292e-002  % m_vals = 3
    2.539398330063230e-001  % m_vals = 5
    9.504178996162932e-001  % m_vals = 7
    2.097847961257068e+000  % m_vals = 9
    5.371920351148152e+000  % m_vals = 13
];

pade_deg = -1;

for i = 1:length(theta)
    if tnorm <= theta(i)
        pade_deg = m_vals(i);
        break;
    end
end

if pade_deg < 0
    % norm(T) too large, need to scale
    [t, s] = log2(tnorm/theta(end));
    s = s - (t == 0.5); % adjust s if normA/theta(end) is a power of 2.
    c = c/2^s;
    r = r/2^s;
    pade_deg = m_vals(end);
else
    % No need to scale
    s = 0;
end

if dbglvl >= 1
    fprintf('S&S parmeters: s=%d  m=%d  norms pre/post scale=%.2e/%.2e\n', ...
        s, pade_deg, tnorm, toeplitz_1norm(c,r));
end

num_coef = diag_pade_coef(pade_deg);
den_coef = num_coef .* ((-1).^(1:(pade_deg+1)));
rat_tic = tic();
[G, B] = toepratval(c,r,num_coef,den_coef);

if dbglvl >= 1
    fprintf('Generator length of rational function: %d', size(G,2));
end

[G,B] = gencompress(G,B);
rat_time = toc(rat_tic);
if dbglvl >= 1
    fprintf(', compressed to %d (%.2fs)\n', size(G,2), rat_time);
end

drank = nan(s+1,1);
drank(1) = size(G, 2);
FF = [];
for k = 1:s
    square_tic = tic;

    % Switch to full matrix representation if drank too large
    if (do_full == true) && isempty(FF) && (drank(k) > size(G,1)/6)
        FF = stein_reconstruction(G, B);
    end
    
    if isempty(FF)
        [G, B] = toeplksquare(G,B);
        lenG_before = size(G,2);
        [G, B] = gencompress(G,B);
        lenG_after = size(G,2);
    else
        FF = FF * FF;
        lenG_before = size(FF,1);
        lenG_after = size(FF,1);
    end
    if dbglvl >= 1
        fprintf('Squaring iter %2d/%2d, generator length %d, compressed to %d (%.2fs)\n',...
            k, s, lenG_before, lenG_after, toc(square_tic));
    end
    drank(k+1) = lenG_after;
end


if do_full && ~isempty(FF)
    G = FF;
    B = [];
end

if dbglvl >= 1
    fprintf('Total time: %.2fs\n', toc(ttotal));
end
end

function c = diag_pade_coef(degree)
% Values taken from Higham
switch degree
    case 3
        c = [120, 60, 12, 1];
    case 5
        c = [30240, 15120, 3360, 420, 30, 1];
    case 7
        c = [17297280, 8648640, 1995840, 277200, 25200, 1512, 56, 1];
    case 9
        c = [17643225600, 8821612800, 2075673600, 302702400, 30270240, ...
            2162160, 110880, 3960, 90, 1
            ];
    case 13
        c = [64764752532480000, 32382376266240000, 7771770303897600, ...
            1187353796428800,   129060195264000,   10559470521600, ...
            670442572800,       33522128640,       1323241920,...
            40840800,           960960,            16380,  182,  1
            ];
end

% Respect Matlab's polyval order.
c = c(end:-1:1);
end


function tnorm = toeplitz_1norm(c,r)
% Compute 1-norm of a given Toeplitz matrix in O(n)

n = length(c);
colnorm = sum(abs(c));
tnorm = colnorm;

for j = 2:n
    colnorm = colnorm - abs(c(end-j+2)) + abs(r(j));
    tnorm = max(tnorm, colnorm);
end
end

function [G, B, drank] = subdiagonal_pade(c, r, dbglvl)
ttotal = tic;
tstamp = tic;
n = length(c);

if dbglvl >= 1
    fprintf('Using scaling&squaring with subdiagonal Pade approximation\n');
end
[est_norm_T, num_iter] = toepnormest(c,r);
if dbglvl >= 1
    fprintf('Estimated norm(T) %.2e, %d iters\n', est_norm_T, num_iter);
    if dbglvl >= 3
        fprintf('True norm(T): %.2e\n', norm(toeplitz(c,r)));
    end
end

[s, k, m] = sexpm_parameters(est_norm_T);
if dbglvl >= 1
    fprintf('Using scaling exponent %d, num degree %d, dnm degree %d\n',...
        s, k, m);
end


if s==0 && k==1 && m==0
    % This means effectively expm(T) = I + T + O(eps)
    c(1) = c(1) + 1;
    r(1) = r(1) + 1;
    [G,B] = stein_generator(c,r);
    drank = 1;
    return;
end

cs = c / (2^s);
rs = r / (2^s);

[resid, poles, remainder] = sexpm_coeffs(k, m);
assert(length(resid) == length(poles));
if dbglvl >= 1
    fprintf('Using pf expansion with %d terms and deg %d remainder\n',...
        length(resid), length(remainder)-1);
end

G = [];
B = [];

e1 = zeros(n,1);
e1(1) = 1.0;

tim = toc(tstamp);
tstamp = tic;
if dbglvl >= 1
    fprintf('Elapsed time for setup: %.2fs\n', tim);
end

if isreal(cs) && isreal(rs)
    % Real Toeplitz matrix
    for j=2:2:length(resid)
        % Take advantage of complex conjugate poles
        [Ginv, Binv] = toepinv_generators(...
            cs - poles(j)*e1, rs - poles(j)*e1);
        Ginv = resid(j) * Ginv;
        G = [G, 2 * real(Ginv), 2 * imag(Ginv)]; %#ok<AGROW>
        B = [B, real(Binv), imag(Binv)]; %#ok<AGROW>
    end
    if mod(length(poles),2)==1
        % Odd degree: Last pole is on the real axis
        [Ginv, Binv] = toepinv_generators(...
            cs - poles(end)*e1, rs - poles(end)*e1);
    
        G = [G, resid(end) * Ginv];
        B = [B, Binv];
    end
else
    % Complex Toeplitz matrix
    for j=1:length(resid)
        % Compute generators for inv(T - poles(j) * eye(n))
        [Ginv, Binv] = toepinv_generators(...
            cs - poles(j)*e1, rs - poles(j)*e1);
    
        % Append generator to list, with residual coefficient on G
        G = [G, resid(j) * Ginv]; %#ok<AGROW>
        B = [B, Binv]; %#ok<AGROW>
    end
end
tim = toc(tstamp);
tstamp = tic();
if dbglvl >= 1
    fprintf('Elapsed time for PF eval: %.2f\n', tim);
end

switch length(remainder)
    case 0
        % Nothing to do
    case 1
        % Add alpha * eye(n) to matrix
        G = [G, remainder(end) * e1];
        B = [B, e1];
    case 2
        % Add beta * T + alpha * eye to matrix
        beta = remainder(end-1);
        alpha = remainder(end);
        [Gadd, Badd] = stein_generator(beta * cs, beta * rs);
        
        % Use special strucuture of canonical stein generators
        Gadd(1,1) = Gadd(1,1) + alpha;
        G = [G, Gadd];
        B = [B, Badd];
    case 3
        % Should use special structure of squared toeplitz matrices.
        % And use case 2 for linear polynomial.
        % FIXME
        error('Implement me');
        
    otherwise
        % Remainder logic for higher degree remainders not implemented
        % FIXME
        error('Broken code path');
end

if dbglvl >= 1
    fprintf('Generator length of pf expansion: %d\n', size(G, 2));
end
if dbglvl >= 2
    fprintf('Generator ranks (G/B): %d/%d\n', rank(G), rank(B));
end


tim = toc(tstamp);
tstamp = tic();
if dbglvl >= 1
    fprintf('Elapsed time for remainder calculation: %.2f\n', tim);
end

[G,B] = gencompress(G,B);
if dbglvl >= 1
    fprintf('Compressing generator...\n');
    fprintf('Generator length of pf expansion: %d\n', size(G, 2));
end
if dbglvl >= 2
    fprintf('Generator ranks (G/B): %d/%d\n', rank(G), rank(B));
end

drank = nan(s+1,1);
drank(1) = size(G,2);
for j=1:s
    itertic = tic();
    [G, B] = toeplksquare(G,B);
    glen1 = size(G,2);
    [G, B] = gencompress(G,B);
    glen2 = size(G,2);
    if dbglvl >= 1
        fprintf('Squaring %d/%d, generator length %d, compressed to %d (%.2fs)\n',...
            j, s, glen1, glen2, toc(itertic));
    end
    drank(j+1) = size(G,2);
end

tim = toc(tstamp);
if dbglvl >= 1
    fprintf('Elapsed time for squaring and compression: %.2f\n', tim);
    fprintf('Generator length after scaling: %d\n', size(G, 2));
end
if dbglvl >= 2
    fprintf('Generator ranks (G/B): %d/%d\n', rank(G), rank(B));
end

tim = toc(ttotal);
if dbglvl >= 1
    fprintf('Total time: %.2fs\n', tim);
end

end % of function

function [s, k, m] = sexpm_parameters(est_norm_T)

if est_norm_T>1
    if est_norm_T      < 200,  s = 4; k = 5; m = k-1;
    elseif  est_norm_T < 1e4,  s = 4; k = 4; m = k+1;
    elseif  est_norm_T < 1e6,  s = 4; k = 3; m = k+1;
    elseif  est_norm_T < 1e9,  s = 3; k = 3; m = k+1;
    elseif  est_norm_T < 1e11, s = 2; k = 3; m = k+1;
    elseif  est_norm_T < 1e12, s = 2; k = 2; m = k+1;
    elseif  est_norm_T < 1e14, s = 2; k = 1; m = k+1;
    else s = 1; k = 1; m = k+1;
    end
else % nrm<1
    if est_norm_T     > .5,   s = 4; k = 4; m = k-1;
    elseif est_norm_T > .3,   s = 3; k = 4; m = k-1;
    elseif est_norm_T > .15,  s = 2; k = 4; m = k-1;
    elseif est_norm_T > .07,  s = 1; k = 4; m = k-1;
    elseif est_norm_T > .01,  s = 0; k = 4; m = k-1;
    elseif est_norm_T > 3e-4, s = 0; k = 3; m = k-1;
    elseif est_norm_T > 1e-5, s = 0; k = 3; m = 0;
    elseif est_norm_T > 1e-8, s = 0; k = 2; m = 0;
    else s = 0; k = 1; m = 0;    % exp(A) = I+A to eps!
    end
end

end

function [r,q,remain] = sexpm_coeffs(k,m) % table of coefficients for each case
% Values taken directly from Guettel/Nakatsukasa 2015.
if m == k+1;
    remain = [];
    if k == 4
        r =  [ -1.582680186458572e+01 - 2.412564578224361e+01i;...
            -1.582680186458572e+01 + 2.412564578224361e+01i;...
            1.499984465975511e+02 + 6.804227952202417e+01i;...
            1.499984465975511e+02 - 6.804227952202417e+01i;
            -2.733432894659307e+02                         ];...
            q = [   3.655694325463550e+00 + 6.543736899360086e+00i;...
            3.655694325463550e+00 - 6.543736899360086e+00i;...
            5.700953298671832e+00 + 3.210265600308496e+00i;...
            5.700953298671832e+00 - 3.210265600308496e+00i;...
            6.286704751729261e+00                        ];
    elseif k==3
        r = [-1.130153999597152e+01 + 1.247167585025031e+01i;...
            -1.130153999597152e+01 - 1.247167585025031e+01i;...
            1.330153999597152e+01 - 6.007173273704750e+01i;...
            1.330153999597152e+01 + 6.007173273704750e+01i];

        q=[3.212806896871536e+00 + 4.773087433276636e+00i;...
            3.212806896871536e+00 - 4.773087433276636e+00i;...
            4.787193103128464e+00 + 1.567476416895212e+00i;...
            4.787193103128464e+00 - 1.567476416895212e+00i];

    elseif k==2
        r=[7.648749087422928e+00 + 4.171640244747463e+00i;...
            7.648749087422928e+00 - 4.171640244747463e+00i;...
            -1.829749817484586e+01                         ];

        q = [2.681082873627756e+00 + 3.050430199247411e+00i;...
            2.681082873627756e+00 - 3.050430199247411e+00i;...
            3.637834252744491e+00           ];
    elseif k==1
        r = [ 1.000000000000000e+00 - 3.535533905932738e+00i;...
            1.000000000000000e+00 + 3.535533905932738e+00i];

        q = [2.000000000000000e+00 + 1.414213562373095e+00i;...
            2.000000000000000e+00 - 1.414213562373095e+00i];
    end
    return
end
if m==k-1,
    if k==5
        r = [     -1.423367961376821e+02 - 1.385465094833037e+01i;...
            -1.423367961376821e+02 + 1.385465094833037e+01i;...
            2.647367961376822e+02 - 4.814394493714596e+02i;...
            2.647367961376822e+02 + 4.814394493714596e+02i];
        q = [      5.203941240131764e+00 + 5.805856841805367e+00i;...
            5.203941240131764e+00 - 5.805856841805367e+00i;...
            6.796058759868242e+00 + 1.886649260140217e+00i;...
            6.796058759868242e+00 - 1.886649260140217e+00i];

        remain =   [2.000000000000000e-01     9.8000000000000e+00];
    elseif k==4
        r = [    2.484269593165883e+01 + 7.460342395992306e+01i;...
            2.484269593165883e+01 - 7.460342395992306e+01i;...
            -1.734353918633177e+02                         ];
        q = [      4.675757014491557e+00 + 3.913489560603711e+00i;...
            4.675757014491557e+00 - 3.913489560603711e+00i;...
            5.648485971016893e+00                         ];
        remain =[    -2.500000000000000e-01    -7.750000000000000e+00        ];
    elseif k==3
        r=  [    2.533333333333333e+01 - 2.733333333333333e+01i;...
            2.533333333333333e+01 + 2.733333333333333e+01i];
        q = [      4.00000000000000e+00 + 2.00000000000000e+00i;...
            4.00000000000000e+00 - 2.00000000000000e+00i];
        remain =[     3.333333333333333e-01     5.666666666666667e+00        ];
    elseif k==2
        r =  -13.5     ;
        q =    3       ;
        remain =[ -0.5  -3.5];
    end
end
if m==0
    r=[];q=[];
    if k==3
        remain = [1/6            1/2            1              1           ];
    elseif k==2
        remain = [   1/2            1              1       ];
    end
end
end
