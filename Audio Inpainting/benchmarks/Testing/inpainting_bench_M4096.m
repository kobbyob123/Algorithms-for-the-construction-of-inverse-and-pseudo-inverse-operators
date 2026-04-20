%% Full-scale benchmark matching inpainting_comparison.m geometric settings:
%   M=4096, F=4096, a=2048
%
% NMF parameters (K=10, maxit=20, nmfit=1) are scaled down from the
% original (K=20, maxit=100, nmfit=10) to keep runtime practical.
% This is sufficient for comparing solvers against each other, but
% condition numbers of per-frame systems may differ from the full setup.
%
% At this scale:
%   - Cholesky: O(m_n^3) with m_n up to ~M/2, ~8x more expensive than M=2048
%   - lsmr_op:  O(F log F) per iter, ~2x more expensive than M=2048
%   - lsmr_matrix: same memory as exact solvers, included for completeness
clear; clc;

% Pre-start parallel pool to avoid startup overhead in timing
p = gcp('nocreate');
if isempty(p)
    parpool('Processes', 4);
end

[y, fs] = audioread('signals/mamavatu.wav');
y  = resample(y, 16e3, fs);
fs = 16e3;
L  = length(y);
rng(0)
mask = rand(L, 1) > 0.5;
K     = 10;
maxit = 20;
nmfit = 1;

% --- Full scale settings -------------------------------------------------
M = 4096;
a = 2048;
F = 4096;
fprintf('M=%d  F=%d  a=%d\n\n', M, F, a);
fprintf('%-15s | %-12s | %-10s | %-8s\n', 'Solver', 'itnlim', 'Time (s)', 'SDR (dB)');
fprintf('%s\n', repmat('-', 1, 55));

% --- Reference solvers (exact) -------------------------------------------
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

% --- LSMR sweep: both matrix and operator variants -----------------------
% Tolerance disabled (atol=0): LSMR runs to exactly itnlim iterations,
% so we isolate the effect of iteration budget on quality and timing.
itnlim_values = [1000, 2000, 5000];
for sv = {'lsmr_matrix', 'lsmr_op'}
    sv = sv{1};
    for itnlim = itnlim_values
        t_start  = tic;
        restored = ainmf_modified('AM', y, mask, K, maxit, ...
            'M', M, 'a', a, 'F', F, 'nmfit', nmfit, ...
            'solver',      sv, ...
            'lsmr_atol',   0, ...       % disabled — sweeping itnlim only
            'lsmr_itnlim', itnlim, ...
            'verbose',     false, ...
            'saveall',     false);
        t_elapsed = toc(t_start);
        gap_idx = find(~mask);
        sdr = 20*log10(norm(y(gap_idx)) / norm(y(gap_idx) - restored(gap_idx)));
        fprintf('%-15s | %-12d | %-10.2f | %-8.2f\n', sv, itnlim, t_elapsed, sdr);
    end
    fprintf('%s\n', repmat('-', 1, 55));
end
