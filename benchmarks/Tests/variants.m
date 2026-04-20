% Standalone validation of all four implementations of line 5 of Algorithm 3
%
% Line 5: s_n = D_n * T^H * M_n^T * (M_n * T * D_n * T^H * M_n^T)^-1 * x_obs_n
%
% Four implementations compared:
%   (1) Matrix + backslash
%   (2) Matrix + Cholesky
%   (3) Matrix + iterative (LSMR)
%   (4) Operator + iterative (LSMR with function handle, no matrix formed)
%
% All four should produce the same s_n up to solver tolerance.

% clear; clc;

% -------------------------------------------------------------------------
%  Problem setup
% -------------------------------------------------------------------------
rng(42);

M = 64;   
F = 128;   
K = 5;     

% The synthesis operator T: F -> M  (IFFT-based)
% ainmf.m: Top = @(x) crop(ifft(x), M) * sqrt(F)
%          Uop = @(x) fft(x, F) / sqrt(F)
Top = @(x) real(ifft(x, F));           
Uop = @(x) fft(x, F);

% Form T as an explicit matrix (F columns, M rows after crop)
% In ainmf.m: Tmat = Top(eye(F))
Tmat = zeros(F, F);
for f = 1:F
    e_f        = zeros(F, 1);
    e_f(f)     = 1;
    Tmat(:, f) = real(ifft(e_f, F));
end
Tmat = Tmat(1:M, :);   % crop to M rows  →  now M x F

% Simulate one STFT frame:
%   mmask   : which of the M samples are observed (logical, M x 1)
%   V_n     : NMF model variances for this frame (F x 1, all positive)
%   x_obs   : observed samples (m_n x 1)

gap_fraction = 0.3;                          % 30% of samples are missing
mmask  = rand(M, 1) > gap_fraction;          % logical mask
m_n    = sum(mmask);                         % number of observed samples
V_n    = abs(randn(F, 1)) + 0.1;            % positive variances

% Ground-truth clean frame (used only to generate realistic x_obs)
x_clean = randn(M, 1);
x_obs   = x_clean(mmask);                   % m_n x 1

fprintf('Problem dimensions:\n');
fprintf('  M (frame length)        = %d\n', M);
fprintf('  F (frequency channels)  = %d\n', F);
fprintf('  m_n (observed samples)  = %d  (%.0f%% observed)\n', ...
        m_n, 100*m_n/M);
fprintf('\n');

% -------------------------------------------------------------------------
%  Build the shared building blocks (used by implementations 1, 2, 3)
%
%  MT   = M_n * T        (m_n x F)  — observed rows of Tmat
%  DTM  = D_n * T^H * M_n^T  (F x m_n)  — scale columns of MT' by V_n
%  A    = M_n*T*D_n*T^H*M_n^T  (m_n x m_n, symmetric positive definite)
% -------------------------------------------------------------------------
MT   = Tmat(mmask, :);          % m_n x F
DTM  = MT' .* V_n;              % F x m_n   (broadcast: each col scaled by V_n)
A    = real(MT * DTM);          % m_n x m_n  (real: exploiting conjugate symmetry)

% -------------------------------------------------------------------------
%  Implementation 1 — Matrix + backslash
% -------------------------------------------------------------------------
z1 = A \ x_obs;
s1 = DTM * z1;

fprintf('Implementation 1 (matrix + backslash):   done\n');

% -------------------------------------------------------------------------
%  Implementation 2 — Matrix + Cholesky
% -------------------------------------------------------------------------
R  = chol(A);                   % R^T * R = A  (upper triangular R)
z2 = R \ (R' \ x_obs);         % forward/back substitution
s2 = DTM * z2;

fprintf('Implementation 2 (matrix + Cholesky):    done\n');

% -------------------------------------------------------------------------
%  Implementation 3 — Matrix + LSMR (iterative, explicit A)
% -------------------------------------------------------------------------
atol    = 1e-9;
btol    = 1e-9;
conlim  = 1e8;
itnlim  = 500;

[z3, ~, ~, itn3] = mod_lsmr(A, x_obs, 0, atol, btol, conlim, itnlim, ...
                              0, false, []);
s3 = DTM * z3;

fprintf('Implementation 3 (matrix + LSMR):        done  (%d iterations)\n', itn3);

% -------------------------------------------------------------------------
%  Implementation 4 — Operator + LSMR (no matrix A formed)
%
%  A_op(v, 1) applies A*v  via: M_n * T * D_n * T^H * M_n^T * v
%  A_op(v, 2) applies A'*v — identical since A is symmetric
%
%  The five steps:
%    1. M_n^T * v       : zero-pad to full M-dim frame
%    2. T^H * result    : analysis (Uop after extending to M)
%    3. D_n * result    : scale by NMF variances V_n
%    4. T * result      : synthesis (Top)
%    5. M_n * result    : select observed rows
% -------------------------------------------------------------------------
A_op = @(v, mode) apply_A(v, mmask, V_n, Tmat, M, F);
% Note: mode argument ignored since A is symmetric (forward = adjoint)

[z4, ~, ~, itn4] = mod_lsmr(A_op, x_obs, 0, atol, btol, conlim, itnlim, ...
                              0, false, []);
s4 = DTM * z4;     % DTM still used for the final multiply (F x m_n, lightweight)

fprintf('Implementation 4 (operator + LSMR):      done  (%d iterations)\n', itn4);

% -------------------------------------------------------------------------
%  Compare all four outputs
% -------------------------------------------------------------------------
fprintf('\n--- Comparison (error relative to implementation 1) ---\n');
fprintf('  ||s2 - s1|| / ||s1||  =  %.2e   (Cholesky vs backslash)\n', ...
        norm(s2 - s1) / norm(s1));
fprintf('  ||s3 - s1|| / ||s1||  =  %.2e   (LSMR matrix vs backslash)\n', ...
        norm(s3 - s1) / norm(s1));
fprintf('  ||s4 - s1|| / ||s1||  =  %.2e   (LSMR operator vs backslash)\n', ...
        norm(s4 - s1) / norm(s1));

fprintf('\n--- Residuals ||A*z - x_obs|| ---\n');
fprintf('  backslash:  %.2e\n', norm(A*z1 - x_obs));
fprintf('  Cholesky:   %.2e\n', norm(A*z2 - x_obs));
fprintf('  LSMR mat:   %.2e\n', norm(A*z3 - x_obs));
fprintf('  LSMR op:    %.2e\n', norm(A*z4 - x_obs));

% =========================================================================
%  Local function: operator apply for implementation 4
%  Applies A = M_n * T * D_n * T^H * M_n^T to vector v (m_n x 1)
%  without forming A explicitly.
% =========================================================================
function result = apply_A(v, obs_mask, d, Tmat, M, F)
    % Step 1: M_n^T * v — zero-pad to full frame (M x 1)
    x_full          = zeros(M, 1);
    x_full(obs_mask) = v;

    % Step 2: T^H * x_full — analysis via FFT (F x 1)
    %   In ainmf.m this would be: Uop(x_full) = fft(x_full, F)/sqrt(F)
    %   Here we use Tmat' which is equivalent
    x_freq = Tmat' * x_full;     % F x 1

    % Step 3: D_n * x_freq — scale by variances (elementwise)
    x_scaled = x_freq .* d;      % F x 1

    % Step 4: T * x_scaled — synthesis (M x 1)
    %   In ainmf.m this would be: Top(x_scaled)
    x_time = real(Tmat * x_scaled);   % M x 1

    % Step 5: M_n * x_time — select observed rows (m_n x 1)
    result = x_time(obs_mask);
end
