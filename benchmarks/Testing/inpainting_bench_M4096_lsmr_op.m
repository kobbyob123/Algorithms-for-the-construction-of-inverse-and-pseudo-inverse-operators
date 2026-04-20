% Full-scale benchmark at M=4096 — lsmr_op only.
% Reference solvers included for comparison baseline.

clear; clc;

addpath(genpath("lsmr\modified\mod_lsmr"));

[y, fs] = audioread('signals/mamavatu.wav');
y  = resample(y, 16e3, fs);
fs = 16e3;
L  = length(y);

rng(0)
mask = rand(L, 1) > 0.5;

K     = 10;
maxit = 1;
nmfit = 1;
M     = 4096;
a     = 2048;
F     = 4096;

fprintf('M=%d  F=%d  a=%d\n\n', M, F, a);
fprintf('%-15s | %-12s | %-10s | %-8s\n', 'Solver', 'itnlim', 'Time (s)', 'SDR (dB)');
fprintf('%s\n', repmat('-', 1, 55));

% --- Reference solvers ---------------------------------------------------
for sv = {'backslash', 'cholesky'}
    sv = sv{1};
    t_start  = tic;
    restored = ainmf_modified('AM', y, mask, K, maxit, ...
        'M', M, 'a', a, 'F', F, 'nmfit', nmfit, ...
        'solver', sv, 'verbose', false, 'saveall', false);
    t_elapsed = toc(t_start);
    gap_idx = find(~mask);
    sdr = 20*log10(norm(y(gap_idx)) / norm(y(gap_idx) - restored(gap_idx)));
    fprintf('%-15s | %-12s | %-10.2f | %-8.2f\n', sv, 'N/A', t_elapsed, sdr);
end

fprintf('%s\n', repmat('-', 1, 55));

% --- lsmr_op sweep -------------------------------------------------------
itnlim_values = [100, 200, 350, 500, 1000, 2000, 5000];

for itnlim = itnlim_values
    t_start  = tic;
    restored = ainmf_modified('AM', y, mask, K, maxit, ...
        'M', M, 'a', a, 'F', F, 'nmfit', nmfit, ...
        'solver',      'lsmr_op', ...
        'lsmr_atol',   0, ...
        'lsmr_itnlim', itnlim, ...
        'verbose',     false, ...
        'saveall',     false);
    t_elapsed = toc(t_start);
    gap_idx = find(~mask);
    sdr = 20 * log10(norm(y(gap_idx)) / norm(y(gap_idx) - restored(gap_idx)));
    fprintf('%-15s | %-12d | %-10.2f | %-8.2f\n', 'lsmr_op', itnlim, t_elapsed, sdr);
end

fprintf('%s\n', repmat('-', 1, 55));
