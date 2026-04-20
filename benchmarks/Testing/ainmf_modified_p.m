% ainmf_modified_2_new one

function [restored, W, H, relnorms, objectives] = ainmf_modified(method, signal, mask, K, maxit, varargin)
% AINMF_MODIFIED — ainmf.m with pluggable line-5 solvers for thesis benchmarking
%
% Adds one new parameter vs the original ainmf.m:
%
%   'solver'   ('backslash')  which implementation to use for line 5:
%              'backslash'    matrix + backslash  (original default)
%              'cholesky'     matrix + Cholesky   (original usechol=true)
%              'lsmr_matrix'  matrix + LSMR       (iterative, explicit A)
%              'lsmr_op'      operator + LSMR     (FFT-based, no matrix A)
%
%   'lsmr_atol'  (1e-6)   LSMR tolerance (atol = btol)
%   'lsmr_itnlim' (300)   LSMR max iterations per frame
%
% All other parameters are identical to ainmf.m.
% The output interface is identical — drop-in replacement.

%% parsing the inputs
pars = inputParser;
pars.KeepUnmatched = true;

addParameter(pars, 'nmfit',    1)
addParameter(pars, 'saveall',  false)
addParameter(pars, 'M',        2048)
addParameter(pars, 'a',        1024)
addParameter(pars, 'F',        2048)
addParameter(pars, 'verbose',  false)
addParameter(pars, 'drawing',  false)
addParameter(pars, 'likelihood', [])
addParameter(pars, 'epsilon',  1e-4)
addParameter(pars, 'tolsol',   0)
addParameter(pars, 'tolobj',   0)
addParameter(pars, 'keepconj', true)
addParameter(pars, 'usegpu',   false)
addParameter(pars, 'usechol',  false)   % kept for backward compatibility

% --- new parameters ---
addParameter(pars, 'solver',       'backslash')
addParameter(pars, 'lsmr_atol',    1e-6)
addParameter(pars, 'lsmr_itnlim', 300)

parse(pars, varargin{:})

nmfit      = pars.Results.nmfit;
saveall    = pars.Results.saveall;
M          = pars.Results.M;
a          = pars.Results.a;
F          = pars.Results.F;
verbose    = pars.Results.verbose;
drawing    = pars.Results.drawing;
likelihood = pars.Results.likelihood;
epsilon    = pars.Results.epsilon;
tolsol     = pars.Results.tolsol;
tolobj     = pars.Results.tolobj;
keepconj   = pars.Results.keepconj;
usegpu     = pars.Results.usegpu;
solver     = pars.Results.solver;
lsmr_atol  = pars.Results.lsmr_atol;
lsmr_itnlim = pars.Results.lsmr_itnlim;

% backward compat: usechol=true maps to solver='cholesky'
if pars.Results.usechol && strcmp(solver, 'backslash')
    solver = 'cholesky';
end

if isnumeric(method)
    switchit = method;
else
    switchit = [];
end

%% operators T and U  (unchanged from ainmf.m)
Uop  = @(x) fft(x, F) / sqrt(F);
Top  = @(x) crop(ifft(x), M) * sqrt(F);
Tmat = Top(eye(F));      % M x F  — used by backslash/cholesky/lsmr_matrix
Umat = Uop(eye(M));
if usegpu
    Tmat = gpuArray(Tmat);
    Umat = gpuArray(Umat);
end

%% initialization  (unchanged from ainmf.m)
L = ceil(length(signal)/a)*a;
N = L/a;
data = [signal; zeros(L-length(signal), 1)];
mask = [mask;   true(L-length(signal),  1)];

rng(0)
W = 0.5*(1.5*abs(randn(F, K))+0.5);
H = 0.5*(1.5*abs(randn(K, N))+0.5);
if usegpu; W = gpuArray(W); H = gpuArray(H); end

if keepconj
    W(floor(F/2)+2:end, :) = flipud(W(2:floor(F/2), :));
end

if saveall
    mrestored_all = zeros(M, maxit, N);
else
    mrestored = zeros(M, N);
end
if usegpu
    if saveall; mrestored_all = gpuArray(mrestored_all);
    else;        mrestored    = gpuArray(mrestored);     end
end

g    = gabwin('sine', a, M, L);
% gana = normalize(g, 'peak'); % deprecated from ltfat
gana = setnorm(g, 'peak');
gana = fftshift(gana);
gsyn = gabdual(gana, a, M)*M;

mdata = NaN(M, N);
mmask = false(M, N);
for n = 1:N
    indices       = 1 + (n-1)*a - floor(M/2) : (n-1)*a + ceil(M/2);
    indices       = 1 + mod(indices-1, L);
    mdata(:, n)   = data(indices) .* gana;
    mmask(:, n)   = mask(indices);
end
if usegpu; mdata = gpuArray(mdata); mmask = gpuArray(mmask); end

UT = Umat*Tmat;

if nargout > 3 || tolsol > 0
    relnorms   = NaN(maxit, 1);
    presolution = NaN(L, 1);
end
if nargout > 4 || tolobj > 0
    objectives = NaN(maxit, 1);
end

V = W*H + epsilon;

%% main iterations
if verbose; str = []; end

for i = 1:maxit

    if ~isempty(switchit)
        if i <= switchit; method = 'AM'; else; method = 'EMtf'; end
    end

    if verbose
        fprintf(repmat('\b', 1, length(str)))
        str = sprintf('%-6s Iteration %d of %d.\n', [method, ':'], i, maxit);
        fprintf('%s', str)
    end

    P = NaN(F, N);

    %----------------------------------------------------------------------
    %  E-step  (only EMtf modified — AM and EMt unchanged)
    %----------------------------------------------------------------------
    if strcmpi(method, 'EMtf')
        % Capture variables needed inside parfor
        solver_     = solver;
        lsmr_atol_  = lsmr_atol;
        lsmr_itnlim_= lsmr_itnlim;
        keepconj_   = keepconj;
        Tmat_       = Tmat;
        Top_        = Top;
        Uop_        = Uop;
        M_          = M;
        F_          = F;

        parfor n = 1:N
            if sum(mmask(:, n)) == M
                % all samples observed — no inversion needed
                s        = Uop_(mdata(:, n));    %#ok<*PFBNS>
                diagSigma = 0;
            else
                % observed rows of T and scaled adjoint
                MT  = Tmat_(mmask(:, n), :);
                DTM = MT' .* V(:, n);

                % system matrix A = M_n * T * D_n * T^H * M_n^T
                MTDTM = MT * DTM;
                if keepconj_
                    MTDTM = real(MTDTM);
                end

                y_obs = mdata(mmask(:, n), n);

                %----------------------------------------------------------
                %  Line 5: solve  MTDTM * z = y_obs,  then  s = DTM * z
                %
                %  Four implementations selectable via 'solver' parameter
                %----------------------------------------------------------
                switch solver_
                    case 'backslash'
                        % Implementation 1 — original default
                        z = MTDTM \ y_obs;

                    case 'cholesky'
                        % Implementation 2 — Cholesky factorisation
                        R = chol(real(MTDTM));
                        z = R \ (R' \ y_obs);

                    case 'lsmr_matrix'
                        % Implementation 3 — LSMR with explicit matrix
                        [z, ~, ~, ~] = mod_lsmr(MTDTM, y_obs, 0, ...
                            lsmr_atol_, lsmr_atol_, 1e8, lsmr_itnlim_, ...
                            0, false, []);

                    case 'lsmr_op'
                        % Implementation 4 — LSMR with FFT operator
                        % A never formed — each mat-vec is O(F log F)
                        A_op = @(v, mode) apply_A_op( ...
                            v, mmask(:,n), V(:,n), Top_, Uop_, M_);
                        [z, ~, ~, ~] = mod_lsmr(A_op, y_obs, 0, ...
                            lsmr_atol_, lsmr_atol_, 1e8, lsmr_itnlim_, ...
                            0, false, []);
                        
                    otherwise
                        error('ainmf_modified: unknown solver ''%s''', solver_);
                end

                % s = D_n * T^H * M_n^T * z  (line 5 result)
                s = DTM * z;

                % posterior variance (exact only for direct solvers)
                % for iterative solvers: set to 0 (AM approximation)
                if strcmp(solver_, 'backslash') || strcmp(solver_, 'cholesky')
                    DTMMTDTM  = DTM / MTDTM;
                    diagSigma = V(:, n) - sum(DTMMTDTM .* conj(DTM), 2);
                else
                    diagSigma = 0;   % iterative: drop variance correction
                end
            end

            if saveall
                mrestored_all(:, i, n) = real(Top_(s));
            else
                mrestored(:, n) = real(Top_(s));
            end

            P(:, n) = abs(s).^2 + abs(diagSigma);
        end
    end

    % EMt and AM branches — unchanged from ainmf.m
    if strcmpi(method, 'EMt')
        parfor n = 1:N
            if sum(mmask(:, n)) == M
                s = Uop(mdata(:, n));
                diagSigma = 0;
            else
                MT = Tmat(mmask(:, n), :);
                DTM = MT'.*V(:, n);
                MTDTM = MT*DTM;
                if keepconj; MTDTM = real(MTDTM); end
                R = chol(real(MTDTM));
                DTMMTDTM = (DTM/R)/R';
                s = Uop(Top(DTMMTDTM*mdata(mmask(:, n), n)));
                diagSigma = sum((Uop(Top(diag(V(:, n)) - DTMMTDTM*DTM'))) .* conj(UT), 2);
            end
            if saveall; mrestored_all(:, i, n) = real(Top(s));
            else;        mrestored(:, n)        = real(Top(s)); end
            P(:, n) = abs(s).^2 + abs(diagSigma);
        end
    end

    if strcmpi(method, 'am')
        % Capture variables for parfor
        solver_      = solver;
        lsmr_atol_   = lsmr_atol;
        lsmr_itnlim_ = lsmr_itnlim;
        keepconj_    = keepconj;
        Tmat_        = Tmat;
        Top_         = Top;
        Uop_         = Uop;
        M_           = M;

        parfor n = 1:N
            if sum(mmask(:, n)) == M
                xhat = mdata(:, n);
                s    = Uop_(xhat);   %#ok<*PFBNS>
            else
                MT    = Tmat_(mmask(:, n), :);
                DTM   = MT' .* V(:, n);
                MTDTM = MT * DTM;
                if keepconj_; MTDTM = real(MTDTM); end

                y_obs = mdata(mmask(:, n), n);

                % Line 5 — solver switch (AM never needs diagSigma)
                switch solver_
                    case 'backslash'
                        z = MTDTM \ y_obs;

                    case 'cholesky'
                        R = chol(real(MTDTM));
                        z = R \ (R' \ y_obs);

                    case 'lsmr_matrix'
                        [z, ~, ~, ~] = mod_lsmr(MTDTM, y_obs, 0, ...
                            lsmr_atol_, lsmr_atol_, 1e8, lsmr_itnlim_, ...
                            0, false, []);
                   

                    case 'lsmr_op'
                        A_op = @(v, mode) apply_A_op( ...
                            v, mmask(:,n), V(:,n), Top_, Uop_, M_);
                        [z, ~, ~, ~] = mod_lsmr(A_op, y_obs, 0, ...
                            lsmr_atol_, lsmr_atol_, 1e8, lsmr_itnlim_, ...
                            0, false, []);

                    otherwise
                        error('ainmf_modified: unknown solver ''%s''', solver_);
                end

                s    = DTM * z;
                xhat = Top_(s);
            end
            if saveall; mrestored_all(:, i, n) = real(xhat);
            else;        mrestored(:, n)        = real(xhat); end
            P(:, n) = abs(s).^2;   % AM: no diagSigma term
        end
    end

    %----------------------------------------------------------------------
    %  M-step — unchanged from ainmf.m
    %----------------------------------------------------------------------
    for j = 1:nmfit
        W = W .* ((V.^(-2) .* P) * H') ./ (V.^(-1) * H');
        V = W*H + epsilon;
        H = H .* (W' * (V.^(-2) .* P)) ./ (W' * V.^(-1));
        V = W*H + epsilon;
        scale = sum(W, 1);
        W = W ./ scale;
        H = H .* scale';
    end

    %----------------------------------------------------------------------
    %  Relative norm, objective, stopping — unchanged from ainmf.m
    %----------------------------------------------------------------------
    if nargout > 3 || tolsol > 0
        if saveall; mrestored = squeeze(mrestored_all(:, i, :)); end
        postsolution = zeros(L, 1);
        for n = 1:N
            indices = 1 + (n-1)*a - floor(M/2) : (n-1)*a + ceil(M/2);
            indices = 1 + mod(indices-1, L);
            postsolution(indices) = postsolution(indices) + mrestored(:, n).*gsyn;
        end
        presolution  = presolution(1:length(signal));
        postsolution = postsolution(1:length(signal));
        relnorms(i)  = norm(postsolution - presolution)/norm(presolution);
        presolution  = postsolution;
    end

    if nargout > 4 || tolobj > 0
        if isempty(likelihood)
            if strcmpi(method, 'am'); likelihood = 'full';
            else;                     likelihood = 'observation'; end
        end
        if strcmpi(likelihood, 'observation')
            objectives(i) = objectiveObservation(V, Tmat, mmask, mdata, keepconj);
        else
            if saveall; mrestored = squeeze(mrestored_all(:, i, :)); end
            objectives(i) = objectiveFull(V, Tmat, mmask, mrestored, keepconj);
        end
        if isnan(objectives(i)); break; end
    end

    if tolsol > 0 && i > 1
        if relnorms(i) < tolsol; break; end
    end
    if tolobj > 0 && i > 1
        if objectives(i) < objectives(i-1)
            if abs(objectives(i)-objectives(i-1))/abs(objectives(i-1)) < tolobj
                break
            end
        end
    end
end

%% overlap-add  (unchanged from ainmf.m)
if saveall
    restored = zeros(L, maxit);
    for n = 1:N
        indices = 1 + (n-1)*a - floor(M/2) : (n-1)*a + ceil(M/2);
        indices = 1 + mod(indices-1, L);
        restored(indices, :) = restored(indices, :) + mrestored_all(:, :, n).*repmat(gsyn, 1, maxit);
    end
else
    restored = zeros(L, 1);
    for n = 1:N
        indices = 1 + (n-1)*a - floor(M/2) : (n-1)*a + ceil(M/2);
        indices = 1 + mod(indices-1, L);
        restored(indices) = restored(indices) + mrestored(:, n).*gsyn;
    end
end

restored = crop(restored, length(signal));
if saveall; restored = restored(:, 1:i); end
if nargout > 3; relnorms   = relnorms(1:i);   end
if nargout > 4; objectives = objectives(1:i); end

end % function ainmf_modified


% =========================================================================
%  apply_A_op — operator apply for lsmr_op solver (task 3)
%
%  Applies A = M_n * T * D_n * T^H * M_n^T  to v using Top/Uop.
%  No matrix ever formed. Each call costs O(F log F).
% =========================================================================
function result = apply_A_op(v, obs_mask, d, Top, Uop, M)
    x_full           = zeros(M, 1);
    x_full(obs_mask) = v;
    x_freq   = Uop(x_full);
    x_scaled = x_freq .* d;
    x_time   = real(Top(x_scaled));
    result   = x_time(obs_mask);
end


% =========================================================================
%  Objective functions — unchanged from ainmf.m
% =========================================================================
function val = objectiveObservation(V, T, mmask, mdata, keepconj)
    N = size(V, 2);
    val = NaN(N, 1);
    parfor n = 1:N
        MT    = T(mmask(:, n), :);
        DTM   = MT'.*V(:, n);
        MTDTM = MT*DTM;
        if keepconj; MTDTM = real(MTDTM);
        else;         MTDTM = (MTDTM + MTDTM')/2; end
        [R, er] = chol(MTDTM);
        if ~er
            val(n) = sum(mmask(:, n)) * log(pi) ...
                   + 2 * sum(log(diag(R))) ...
                   + mdata(mmask(:, n), n)'*(R\(R'\mdata(mmask(:, n), n)));
        end
    end
    val = real(sum(val));
end

function val = objectiveFull(V, T, mmask, mrestored, keepconj)
    N = size(V, 2);
    val = NaN(N, 1);
    parfor n = 1:N
        xhat = mrestored(:, n);
        TDT  = T*(T'.*V(:, n));
        if keepconj; TDT = real(TDT);
        else;         TDT = (TDT + TDT')/2; end
        [R, er] = chol(TDT);
        if ~er
            val(n) = length(mmask(:, n)) * log(pi) ...
                   + 2 * sum(log(diag(R))) ...
                   + xhat'*(R\(R'\xhat));
        end
    end
    val = real(sum(val));
end


% =========================================================================
%  crop — inlined from utils/crop.m so ainmf_modified has no path deps
%  Crops input array x to length N along the first dimension.
% =========================================================================
function y = crop(x, N)
    oldsz = size(x);
    oldN  = oldsz(1);
    x     = reshape(x, oldN, []);
    if N < oldN
        y = x(1:N, :);
    elseif N > oldN
        y = [x; zeros(N-oldN, size(x,2))];
    else
        y = x;
    end
    sz    = oldsz;
    sz(1) = N;
    y     = reshape(y, sz);
end
