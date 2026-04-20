%% line5_inpainting_bench.m
% End-to-end benchmark: tests effect of itnlim on SDR and timing.
% atol=0 (not the limiting factor), itnlim swept across values.
% Backslash and Cholesky run once as reference.

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
M     = 1024;
a     = 512;
F     = 2048;

fprintf('%-15s | %-12s | %-10s | %-8s\n', 'Solver', 'itnlim', 'Time (s)', 'SDR (dB)');
fprintf('%s\n', repmat('-', 1, 55));

% --- Reference solvers (run once) ----------------------------------------
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

% --- LSMR: sweep itnlim --------------------------------------------------
itnlim_values = [500, 1000, 2000, 5000];

for sv = {'lsmr_matrix', 'lsmr_op'}
    sv = sv{1};
    for itnlim = itnlim_values
        t_start  = tic;
        restored = ainmf_modified('AM', y, mask, K, maxit, ...
            'M', M, 'a', a, 'F', F, 'nmfit', nmfit, ...
            'solver',       sv, ...
            'lsmr_atol',    0, ...
            'lsmr_itnlim',  itnlim, ...
            'verbose',      false, ...
            'saveall',      false);
        t_elapsed = toc(t_start);
        gap_idx = find(~mask);
        sdr = 20*log10(norm(y(gap_idx)) / norm(y(gap_idx) - restored(gap_idx)));
        fprintf('%-15s | %-12d | %-10.2f | %-8.2f\n', sv, itnlim, t_elapsed, sdr);
    end
    fprintf('%s\n', repmat('-', 1, 55));
end
