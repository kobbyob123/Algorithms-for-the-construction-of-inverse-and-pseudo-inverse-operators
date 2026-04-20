%% line5_variants.m
%  Supervisor task: Isolate line 5 of Algorithm 3 (AM method)
%  Implements 4 variants of the per-frame linear solve:
%    1. Matrix + backslash  (mldivide)
%    2. Matrix + Cholesky   (original ainmf approach)
%    3. Matrix + LSMR       (iterative, explicit matrix)
%    4. Operator + LSMR     (iterative, function handles only)
%
%  Line 5:  s_hat_n = D_n * T^H * M_n^T * (M_n * T * D_n * T^H * M_n^T)^{-1} * x_obs_n
%
%  In the code notation from ainmf.m:
%    MT    = Tmat(mmask(:,n), :)          ... M_n * T  (rows of T selected by mask)
%    DTM   = MT' .* V(:,n)               ... D_n * T^H * M_n^T  (column-scaled adjoint)
%    MTDTM = MT * DTM                    ... M_n * T * D_n * T^H * M_n^T  (the matrix to invert)
%    The solve is:  z = MTDTM \ x_obs_n,  then  s = DTM * z

% clc
clear; close all;
rng(42);

%% === SETUP: Simulate one frame of the inpainting problem ===
F = 512;   % number of frequency channels
M_frame = 256;  % frame length in time domain (< F for oversampled transform)

% Create a random synthesis transform T (M_frame x F)
% In ainmf.m: Tmat = Top(eye(F)) which is M x F (time x frequency)
% For testing we use a random matrix with controlled singular values
[U_t, ~, ~] = svd(randn(M_frame, M_frame), 'econ');  % M_frame x M_frame
[V_t, ~, ~] = svd(randn(F, M_frame), 'econ');         % F x M_frame
s_diag = linspace(1, 0.5, M_frame);  % moderate condition number (kappa = 2)
Tmat = U_t * diag(s_diag) * V_t';   % M_frame x F (same as ainmf.m)

% Simulate a mask: keep ~70% of time-domain samples (30% gap)
mask_ratio = 0.7;
mmask = sort(randperm(M_frame, round(mask_ratio * M_frame)))';

% NMF variance vector for this frame (positive, from V = W*H)
K = 10;
W = abs(randn(F, K)) + 0.01;
H = abs(randn(K, 1)) + 0.01;
v_n = W * H;  % F x 1, the n-th column of V = WH

% Generate a "true" TF coefficient vector and observed samples
s_true = randn(F, 1) + 1i*randn(F, 1);
x_full = real(Tmat * s_true);    % T * s: frequency→time (M_frame x 1)
x_obs  = x_full(mmask);          % observed (masked) samples

%% === BUILD THE MATRICES (shared across variants 1-3) ===
MT    = Tmat(mmask, :);           % M_n * T:  selected rows of T
DTM   = MT' .* v_n;              % D_n * T^H * M_n^T  (F x m_obs)
MTDTM = MT * DTM;               % M_n * T * D_n * T^H * M_n^T  (m_obs x m_obs)
MTDTM = real(MTDTM);            % enforce real (as in ainmf.m with keepconj)

fprintf('Problem size: F=%d, M=%d, observed=%d\n', F, M_frame, length(mmask));
fprintf('Condition number of MTDTM: %.2f\n', cond(MTDTM));
fprintf('---\n');

%% === VARIANT 1: Matrix + Backslash ===
tic;
z1 = MTDTM \ x_obs;
s1 = DTM * z1;
t1 = toc;
fprintf('Variant 1 (matrix + backslash):  time = %.4f s\n', t1);

%% === VARIANT 2: Matrix + Cholesky ===
tic;
R_chol = chol(MTDTM);           % upper triangular: MTDTM = R'*R
z2 = R_chol \ (R_chol' \ x_obs); % solve R'*R*z = x_obs
s2 = DTM * z2;
t2 = toc;
fprintf('Variant 2 (matrix + Cholesky):   time = %.4f s\n', t2);
fprintf('  ||s2 - s1|| = %.2e\n', norm(s2 - s1));

%% === VARIANT 3: Matrix + LSMR (iterative) ===
tol_lsmr = 1e-10;
itnlim_lsmr = 500;

tic;
% LSMR solves min ||MTDTM * z - x_obs||
% We pass the explicit matrix to mod_lsmr
[z3, ~, ~, itn3] = mod_lsmr(MTDTM, x_obs, 0, tol_lsmr, tol_lsmr, 1e7, itnlim_lsmr, 0, false, z1);
s3 = DTM * z3;
t3 = toc;
fprintf('Variant 3 (matrix + LSMR):       time = %.4f s, iters = %d\n', t3, itn3);
fprintf('  ||s3 - s1|| = %.2e\n', norm(s3 - s1));

%% === VARIANT 4: Operator + LSMR (function handles, no matrix stored) ===
% This is the key innovation: we never form MTDTM as a matrix.
% Instead, we define forward and adjoint operators.
%
% The system to solve is:  (M_n * T * D_n * T^H * M_n^T) z = x_obs
%
% Forward operator A(z):
%   1. w = M_n^T * z          ... expand from observed indices to full frame
%   2. u = T^H * w            ... apply adjoint of T  (= Tmat' * w)
%   3. d = D_n * u            ... pointwise multiply by v_n
%   4. q = T * d              ... apply T  (= Tmat * d, but only rows in mask)
%   5. result = M_n * q       ... select observed indices
%
% Since LSMR needs both A and A^T as function handles:
%   A^T is the same chain in reverse (it's self-adjoint for real MTDTM).

% Operator-only forward: MTDTM * z without forming the matrix
Afun_forward = @(z) real(Tmat(mmask,:) * (v_n .* (Tmat(mmask,:)' * z)));

% For LSMR, we pass a function handle that distinguishes forward/adjoint
% mod_lsmr expects: A*x when called with 2nd arg, A'*y when called with 3rd arg
% But our mod_lsmr accepts either a matrix or function handle.
% For function handle mode, it expects A(x, mode): mode=1 for A*x, mode=2 for A'*x
% Since MTDTM is symmetric, both modes do the same thing.

Afun = @(z, mode) Afun_forward(z);  % symmetric, so A = A^T

tic;
[z4, ~, ~, itn4] = mod_lsmr(Afun, x_obs, 0, tol_lsmr, tol_lsmr, 1e7, itnlim_lsmr, 0, false, z1);
% Recover s using operator (no DTM matrix needed)
s4 = v_n .* (Tmat(mmask,:)' * z4);
t4 = toc;
fprintf('Variant 4 (operator + LSMR):     time = %.4f s, iters = %d\n', t4, itn4);
fprintf('  ||s4 - s1|| = %.2e\n', norm(s4 - s1));

%% === SUMMARY TABLE ===
fprintf('\n=== SUMMARY ===\n');
fprintf('%-30s  %10s  %10s  %10s\n', 'Variant', 'Time (s)', 'Iters', '||s - s_bs||');
fprintf('%-30s  %10.4f  %10s  %10s\n', '1. Matrix + Backslash', t1, '-', '-');
fprintf('%-30s  %10.4f  %10s  %10.2e\n', '2. Matrix + Cholesky', t2, '-', norm(s2-s1));
fprintf('%-30s  %10.4f  %10d  %10.2e\n', '3. Matrix + LSMR', t3, itn3, norm(s3-s1));
fprintf('%-30s  %10.4f  %10d  %10.2e\n', '4. Operator + LSMR', t4, itn4, norm(s4-s1));

%% === VERIFY: Compare with direct pseudoinverse for correctness ===
s_pinv = DTM * (pinv(MTDTM) * x_obs);
fprintf('\nVerification against pinv:\n');
fprintf('  ||s1 - s_pinv|| = %.2e\n', norm(s1 - s_pinv));
fprintf('  ||s4 - s_pinv|| = %.2e\n', norm(s4 - s_pinv));