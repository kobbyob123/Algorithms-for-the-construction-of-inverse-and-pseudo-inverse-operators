%% line5_inpainting_bench.m
% End-to-end benchmark of the four line-5 solver implementations
% inside the full NMF inpainting algorithm (ainmf_modified.m).
%
% Measures per-solver:
%   - wall-clock time for the full inpainting run
%   - SDR of the restored signal (gap samples only)

clear; clc;

% Pre-start parallel pool to avoid startup overhead in timing
p = gcp('nocreate');
if isempty(p)
    parpool('Processes', 4);
end

% -------------------------------------------------------------------------
%  Signal and gap setup  (mirrors inpainting_demo.m small settings)
% -------------------------------------------------------------------------
[y, fs] = audioread('signals/mamavatu.wav');
y  = resample(y, 16e3, fs);
fs = 16e3;
L  = length(y);

rng(0)
mask = rand(L, 1) > 0.5;   % ~50% missing — challenging gap scenario

% -------------------------------------------------------------------------
%  Common NMF parameters
% -------------------------------------------------------------------------
K     = 10;
maxit = 20;    % keep low for benchmarking — increase for quality runs - 20
nmfit = 1;
M     = 1024;
a     = 512;
F     = 2048;

% -------------------------------------------------------------------------
%  Solvers to compare
% -------------------------------------------------------------------------
solvers = {'backslash', 'cholesky', 'lsmr_matrix', 'lsmr_op'};

fprintf('%-15s | %-10s | %-8s\n', 'Solver', 'Time (s)', 'SDR (dB)');
fprintf('%s\n', repmat('-', 1, 40));

for s = 1:length(solvers)
    sv = solvers{s};

    t_start  = tic;
    restored = ainmf_modified('AM', y, mask, K, maxit, ...
        'M', M, 'a', a, 'F', F, ...
        'nmfit', nmfit, ...
        'solver',      sv, ...
        'lsmr_atol',   0, ...
        'lsmr_itnlim', 500, ...
        'verbose',     false, ...
        'saveall',     false);
    t_elapsed = toc(t_start);

    % SDR over gap samples only
    gap_idx = find(~mask);
    sdr = 20 * log10( norm(y(gap_idx)) / norm(y(gap_idx) - restored(gap_idx)) );

    fprintf('%-15s | %-10.2f | %-8.2f\n', sv, t_elapsed, sdr);
end
