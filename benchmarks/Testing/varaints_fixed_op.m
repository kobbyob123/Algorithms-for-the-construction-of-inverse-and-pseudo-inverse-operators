%% line5_test_v2.m
% Standalone validation of all four implementations of line 5 of Algorithm 3
%
% Line 5: s_n = D_n * T^H * M_n^T * (M_n * T * D_n * T^H * M_n^T)^-1 * x_obs_n
%
% Four implementations compared:
%   (1) Matrix + backslash
%   (2) Matrix + Cholesky
%   (3) Matrix + iterative (LSMR)
%   (4) Operator + iterative (LSMR — Top/Uop used directly, no Tmat inside)
%
% All four should produce the same s_n up to solver tolerance.

clear; clc;

% -------------------------------------------------------------------------
%  Problem setup
% -------------------------------------------------------------------------
rng(0);

M = 1024;    % window length
F = 2048;   % frequency channels

% --- Define T as actual operators (matching ainmf.m exactly) -------------
%
%   ainmf.m:  Top = @(x) crop(ifft(x), M) * sqrt(F)   — NO real() inside
%             Uop = @(x) fft(x, F) / sqrt(F)
%
% Top: synthesis  F -> M  (IFFT, crop to M rows, scale) — complex output
% Uop: analysis   M -> F  (FFT to F bins, scale)        — complex output
%
% NOTE: real() is NOT part of Top. In ainmf.m, real() is applied only
% when writing the restored signal:  mrestored = real(Top(s))
% Putting real() inside Top makes Tmat real, which breaks the adjoint
% relationship with Uop (which returns complex). That inconsistency
% causes implementation 4 to compute the wrong operator entirely.
%
crop_M = @(x) x(1:M);                           % helper: crop to M rows
Top    = @(x) crop_M(ifft(x, F) * sqrt(F));     % F x 1 -> M x 1 (complex)
Uop    = @(x) fft(x, F) / sqrt(F);              % M x 1 -> F x 1 (complex)

% --- Build Tmat from Top (matches ainmf.m: Tmat = Top(eye(F))) -----------
% Used only by implementations 1, 2, 3 — NOT passed into apply_A
Tmat = zeros(M, F);
for f = 1:F
    e_f       = zeros(F, 1);
    e_f(f)    = 1;
    Tmat(:,f) = Top(e_f);      % complex column — consistent with Uop adjoint
end

% --- Simulate one STFT frame ---------------------------------------------
gap_fraction = 0.3;
mmask  = rand(M, 1) > gap_fraction;   % logical mask, M x 1
m_n    = sum(mmask);
V_n    = abs(randn(F, 1)) + 0.1;     % NMF variances, F x 1, all positive
x_obs  = randn(m_n, 1);              % observed samples, m_n x 1

fprintf('Problem dimensions:\n');
fprintf('  M (frame length)       = %d\n', M);
fprintf('  F (frequency channels) = %d\n', F);
fprintf('  m_n (observed samples) = %d  (%.0f%% observed)\n', m_n, 100*m_n/M);
fprintf('\n');

% -------------------------------------------------------------------------
%  Shared building blocks for implementations 1, 2, 3
%
%  MT   = M_n * T            (m_n x F)   observed rows of Tmat
%  DTM  = D_n * T^H * M_n^T  (F x m_n)   cols of MT' scaled by V_n
%  A    = M_n*T*D_n*T^H*M_n^T (m_n x m_n, symmetric positive definite)
%
%  real() applied to A matches ainmf.m's keepconj=true behaviour:
%  for real audio signals the imaginary part of MTDTM is numerical noise
% -------------------------------------------------------------------------
MT  = Tmat(mmask, :);        % m_n x F  (complex)
DTM = MT' .* V_n;            % F x m_n  (complex)
A   = real(MT * DTM);        % m_n x m_n (real SPD — matches ainmf.m)

% -------------------------------------------------------------------------
%  Implementation 1 — Matrix + backslash
% -------------------------------------------------------------------------
z1 = A \ x_obs;
s1 = DTM * z1;
fprintf('Implementation 1 (matrix + backslash):          done\n');

% -------------------------------------------------------------------------
%  Implementation 2 — Matrix + Cholesky
% -------------------------------------------------------------------------
R  = chol(A);
z2 = R \ (R' \ x_obs);
s2 = DTM * z2;
fprintf('Implementation 2 (matrix + Cholesky):           done\n');

% -------------------------------------------------------------------------
%  Implementation 3 — Matrix + LSMR
% -------------------------------------------------------------------------
atol   = 0;
btol   = atol;
conlim = 1e8;
itnlim = 500;

[z3, ~, ~, itn3] = mod_lsmr(A, x_obs, 0, atol, btol, conlim, itnlim, 0, false, []);
s3 = DTM * z3;
fprintf('Implementation 3 (matrix + LSMR):               done  (%d iters)\n', itn3);

% -------------------------------------------------------------------------
%  Implementation 4 — Operator + LSMR
%
%  Key differences from implementations 1-3:
%    - A is never formed as a matrix
%    - apply_A receives Top and Uop (function handles), not Tmat
%    - each LSMR matrix-vector product costs O(F log F) via FFT
%      instead of O(m_n * F) for an explicit matrix multiply
%
%  A_op accepts (v, mode) to satisfy mod_lsmr's function handle interface.
%  Since A is symmetric, both modes apply the same operator.
% -------------------------------------------------------------------------
A_op = @(v, mode) apply_A(v, mmask, V_n, Top, Uop, M);

[z4, ~, ~, itn4] = mod_lsmr(A_op, x_obs, 0, atol, btol, conlim, itnlim, 0, false, []);

% DTM still needed for the final s = D_n * T^H * M_n^T * z
% (lightweight F x m_n multiply — not the bottleneck)
s4 = DTM * z4;
fprintf('Implementation 4 (operator + LSMR, FFT-based):  done  (%d iters)\n', itn4);

% -------------------------------------------------------------------------
%  Results
% -------------------------------------------------------------------------
fprintf('\n--- Error relative to backslash ---\n');
fprintf('  ||s2 - s1|| / ||s1||  =  %.2e   (Cholesky)\n',      norm(s2-s1)/norm(s1));
fprintf('  ||s3 - s1|| / ||s1||  =  %.2e   (LSMR matrix)\n',   norm(s3-s1)/norm(s1));
fprintf('  ||s4 - s1|| / ||s1||  =  %.2e   (LSMR operator)\n', norm(s4-s1)/norm(s1));

fprintf('\n--- Residuals ||A*z - x_obs|| ---\n');
fprintf('  backslash:      %.2e\n', norm(A*z1 - x_obs));
fprintf('  Cholesky:       %.2e\n', norm(A*z2 - x_obs));
fprintf('  LSMR matrix:    %.2e\n', norm(A*z3 - x_obs));
fprintf('  LSMR operator:  %.2e\n', norm(A*z4 - x_obs));


% =========================================================================
%  apply_A — operator application for implementation 4
%
%  Applies A = M_n * T * D_n * T^H * M_n^T  to vector v (m_n x 1)
%  using Top and Uop directly — Tmat is never formed or used here.
%
%  Five steps mapping back to line 5 of Algorithm 3:
%    1. M_n^T * v    : zero-pad observed vector to full M-dim frame
%    2. T^H * result : analysis  via Uop (FFT-based, O(F log F))
%    3. D_n * result : scale by NMF variances V_n  (elementwise)
%    4. T  * result  : synthesis via Top (IFFT-based, O(F log F))
%    5. M_n * result : select observed rows
%
%  real() at step 4: for real signals the IFFT output is real up to
%  numerical noise. Applying real() keeps the result consistent with
%  the real SPD matrix A used in implementations 1-3.
% =========================================================================
function result = apply_A(v, obs_mask, d, Top, Uop, M)
    % Step 1: M_n^T * v — zero-pad to full M-dim frame
    x_full           = zeros(M, 1);
    x_full(obs_mask) = v;

    % Step 2: T^H * x_full — analysis
    x_freq   = Uop(x_full);              % F x 1, complex

    % Step 3: D_n * x_freq — scale by variances
    x_scaled = x_freq .* d;             % F x 1, complex

    % Step 4: T * x_scaled — synthesis
    x_time   = real(Top(x_scaled));     % M x 1, real
    %   real() here mirrors ainmf.m's keepconj=true behaviour:
    %   imaginary part is numerical noise for real-signal inpainting

    % Step 5: M_n * x_time — select observed rows
    result   = x_time(obs_mask);        % m_n x 1, real
end