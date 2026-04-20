%   1. Measure time (size sweep with timeit)
%   2. Check if LSMR tol = 0 leads to convergence (numerically zero residual)
%   3. apply_A using fft/ifft/masking only — no Tmat inside
%   3.1 Explicit consistency check: norm(apply_A(x) - A*x) = numerical 0

clear; clc;

% =========================================================================
%  SETUP — shared across all tasks
% =========================================================================
rng(42);

M = 64;     % window length
F = 128;    % frequency channels

% Operators matching ainmf.m exactly:
%   Top = @(x) crop(ifft(x), M) * sqrt(F)   synthesis F -> M
%   Uop = @(x) fft(x, F) / sqrt(F)          analysis  M -> F

crop_M = @(x) x(1:M);
Top    = @(x) crop_M(ifft(x, F) * sqrt(F));
Uop    = @(x) fft(x, F) / sqrt(F);

% Tmat built from Top — used ONLY by implementations 1/2/3 and task 3.1
Tmat = zeros(M, F);
for f = 1:F
    e = zeros(F,1); e(f) = 1;
    Tmat(:,f) = Top(e);
end

% Simulate one frame
gap_fraction = 0.3;
mmask = rand(M,1) > gap_fraction;
m_n   = sum(mmask);
V_n   = abs(randn(F,1)) + 0.1;
x_obs = randn(m_n,1);

% Building blocks for matrix implementations
MT  = Tmat(mmask,:);
DTM = MT' .* V_n;
A   = real(MT * DTM);   % m_n x m_n, real SPD

fprintf('================================================\n');
fprintf('  M=%d  F=%d  m_n=%d  (%.0f%% observed)\n', M, F, m_n, 100*m_n/M);
fprintf('================================================\n\n');


% =========================================================================
%  TASK 3.1 — Operator consistency check
%  For a random x, verify: norm(apply_A(x) - A*x) = numerical zero
% =========================================================================
fprintf('--- Task 3.1: Operator consistency check ---\n');

n_checks = 5;
fprintf('  Testing %d random vectors x:\n', n_checks);
for k = 1:n_checks
    x_test       = randn(m_n, 1);
    Ax_matrix    = A * x_test;
    Ax_operator  = apply_A(x_test, mmask, V_n, Top, Uop, M);
    consistency  = norm(Ax_operator - Ax_matrix) / norm(Ax_matrix);
    fprintf('    x_%d:  norm(apply_A(x) - A*x) / norm(A*x)  =  %.2e', k, consistency);
    if consistency < 1e-10
        fprintf('  PASS\n');
    else
        fprintf('  FAIL\n');
    end
end
fprintf('\n');


% =========================================================================
%  TASK 2 — Does LSMR with tol = 0 converge to machine precision?
%  Set atol = btol = 0, conlim = 0 (disabled), let LSMR run until its own
%  internal stopping criteria are met.
% =========================================================================
fprintf('--- Task 2: LSMR with tolerance = 0 ---\n');

[z_tol0, ~, istop, itn_tol0] = mod_lsmr(A, x_obs, 0, 0, 0, 0, 1000, 0, false, []);
s_tol0    = DTM * z_tol0;
res_tol0  = norm(A*z_tol0 - x_obs);

% Reference: backslash
z_ref = A \ x_obs;
s_ref = DTM * z_ref;

fprintf('  Iterations:          %d\n',    itn_tol0);
fprintf('  LSMR stop code:      %d\n',    istop);
fprintf('  Residual ||Az-b||:   %.2e\n',  res_tol0);
fprintf('  ||s_tol0 - s_ref||/||s_ref||: %.2e\n', norm(s_tol0-s_ref)/norm(s_ref));
fprintf('  (backslash residual: %.2e for comparison)\n', norm(A*z_ref - x_obs));
fprintf('\n');


% =========================================================================
%  TASK 1 + 3 — Time comparison across sizes
%  All four implementations, proper operators, FFT-based apply_A
% =========================================================================
fprintf('--- Task 1: Timing across sizes ---\n\n');

configs = [
     64,   128;
    512,  1024;
   1024,  2048;
   2048,  4096;
];

atol   = 1e-9;
btol   = 1e-9;
conlim = 1e8;
itnlim = 2000;

fprintf('%-6s %-6s %-5s | %-8s %-8s %-8s %-10s | %-9s %-9s %-9s | %-5s %-5s\n', ...
    'M','F','m_n','t_bsl','t_chol','t_ls3','t_ls4(op)', ...
    'err_chol','err_ls3','err_ls4','itn3','itn4');
fprintf('%s\n', repmat('-',1,100));

for c = 1:size(configs,1)
    Mc = configs(c,1);
    Fc = configs(c,2);
    rng(42);

    % Build operators and Tmat for this size
    crop_Mc = @(x) x(1:Mc);
    Top_c   = @(x) crop_Mc(ifft(x, Fc) * sqrt(Fc));
    Uop_c   = @(x) fft(x, Fc) / sqrt(Fc);

    Tmat_c = zeros(Mc, Fc);
    for f = 1:Fc
        e = zeros(Fc,1); e(f) = 1;
        Tmat_c(:,f) = Top_c(e);
    end

    mmask_c = rand(Mc,1) > gap_fraction;
    m_n_c   = sum(mmask_c);
    V_n_c   = abs(randn(Fc,1)) + 0.1;
    x_obs_c = randn(m_n_c,1);

    MT_c  = Tmat_c(mmask_c,:);
    DTM_c = MT_c' .* V_n_c;
    A_c   = real(MT_c * DTM_c);

    % Impl 1: backslash
    t1 = timeit(@() A_c \ x_obs_c);
    z1 = A_c \ x_obs_c;  s1 = DTM_c * z1;

    % Impl 2: Cholesky
    t2 = timeit(@() chol_solve(A_c, x_obs_c));
    z2 = chol_solve(A_c, x_obs_c);  s2 = DTM_c * z2;

    % Impl 3: matrix + LSMR
    t3 = timeit(@() mod_lsmr(A_c, x_obs_c, 0, atol, btol, conlim, itnlim, 0, false, []));
    [z3,~,~,itn3] = mod_lsmr(A_c, x_obs_c, 0, atol, btol, conlim, itnlim, 0, false, []);
    s3 = DTM_c * z3;

    % Impl 4: operator + LSMR (FFT)
    A_op_c = @(v, mode) apply_A(v, mmask_c, V_n_c, Top_c, Uop_c, Mc);
    t4 = timeit(@() mod_lsmr(A_op_c, x_obs_c, 0, atol, btol, conlim, itnlim, 0, false, []));
    [z4,~,~,itn4] = mod_lsmr(A_op_c, x_obs_c, 0, atol, btol, conlim, itnlim, 0, false, []);
    s4 = DTM_c * z4;

    e2 = norm(s2-s1)/norm(s1);
    e3 = norm(s3-s1)/norm(s1);
    e4 = norm(s4-s1)/norm(s1);

    fprintf('%-6d %-6d %-5d | %-8.4f %-8.4f %-8.4f %-10.4f | %-9.2e %-9.2e %-9.2e | %-5d %-5d\n', ...
        Mc, Fc, m_n_c, t1, t2, t3, t4, e2, e3, e4, itn3, itn4);
end

fprintf('\nNote: t_ls4(op) uses FFT/IFFT — O(F log F) per iter vs O(m_n * F) for matrix.\n');
fprintf('      Crossover expected around M=512-1024 where operator overtakes Cholesky.\n');


% =========================================================================
%  apply_A — Task 3: operator implementation, no Tmat
%
%  Applies A = M_n * T * D_n * T^H * M_n^T  to v (m_n x 1)
%  using Top (synthesis) and Uop (analysis) directly.
%
%  Steps:
%    1. M_n^T * v   : zero-pad to M-dim frame
%    2. T^H * result : Uop — analysis (FFT)         O(F log F)
%    3. D_n * result : elementwise scale by V_n     O(F)
%    4. T  * result  : Top — synthesis (IFFT+crop)  O(F log F)
%    5. M_n * result : select observed rows         O(m_n)
% =========================================================================
function result = apply_A(v, obs_mask, d, Top, Uop, M)
    % Step 1: M_n^T * v
    x_full           = zeros(M,1);
    x_full(obs_mask) = v;

    % Step 2: T^H
    x_freq   = Uop(x_full);

    % Step 3: D_n
    x_scaled = x_freq .* d;

    % Step 4: T  (real() discards numerical imaginary noise)
    x_time   = real(Top(x_scaled));

    % Step 5: M_n
    result   = x_time(obs_mask);
end


% =========================================================================
%  Helper: Cholesky solve (factored out so timeit captures both steps)
% =========================================================================
function z = chol_solve(A, b)
    R = chol(A);
    z = R \ (R' \ b);
end
