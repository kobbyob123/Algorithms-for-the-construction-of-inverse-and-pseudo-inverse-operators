%% line5_inpainting_bench_M2048.m
% Same experiment as before but at M=2048 (full scale from inpainting_comparison.m)
%
% At M=2048 the exact solve cost is O(m_n^3) with m_n ~ 1400
% vs O(F log F) per LSMR iteration — crossover expected here.

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
M = 2048;
a = 1024;
F = 4096;

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

% --- LSMR sweep ----------------------------------------------------------
% At M=2048 standalone benchmark showed lsmr_op needs only ~36 iterations
% to match exact solvers. In the full pipeline condition numbers are higher
% so we test a range.
itnlim_values = [500, 1000, 2000, 5000];

for sv = {'lsmr_matrix', 'lsmr_op'}
    sv = sv{1};
    for itnlim = itnlim_values
        t_start  = tic;
        restored = ainmf_modified('AM', y, mask, K, maxit, ...
            'M', M, 'a', a, 'F', F, 'nmfit', nmfit, ...
            'solver',      sv, ...
            'lsmr_atol',   0, ...
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