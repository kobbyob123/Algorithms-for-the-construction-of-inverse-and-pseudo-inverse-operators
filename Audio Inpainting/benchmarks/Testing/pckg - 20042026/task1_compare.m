%% task1_compare.m
% Task 1 (Supervisor): Test precision of lsmr_op solution inside ainmf.
%
% Uses solver='compare' mode in ainmf_modified.m, which computes ALL
% four solvers per frame and reports ||z_lsmr_op - z_backslash|| / ||z_backslash||
% and ||s_lsmr_op - s_backslash|| / ||s_backslash|| at every AM iteration.
%
% Two sub-experiments:
%   A) maxit sweep (1, 2, 5, 10, 20) — tests amplification hypothesis:
%      does per-frame error stay small, or compound across iterations?
%   B) single run with verbose compare output at default maxit=20

clear; clc;

[y, fs] = audioread('signals/mamavatu.wav');
y  = resample(y, 16e3, fs);
fs = 16e3;
L  = length(y);

rng(0)
mask = rand(L, 1) > 0.5;

K     = 10;
nmfit = 1;
M     = 1024;
a     = 512;
F     = 2048;

% LSMR settings — use high itnlim so LSMR can fully converge per frame
lsmr_itnlim = 2000;
lsmr_atol   = 0;

% -------------------------------------------------------------------------
%  Sub-experiment A: maxit sweep
%  If per-frame error is numerically zero, SDR should match backslash
%  at maxit=1 and diverge only at higher maxit (amplification hypothesis).
%  If SDR differs even at maxit=1, the error is not zero per frame.
% -------------------------------------------------------------------------
fprintf('=== Sub-experiment A: maxit sweep ===\n');
fprintf('Running backslash and lsmr_op at each maxit value.\n\n');

maxit_values = [1, 2, 5, 10, 20];
gap_idx = find(~mask);

fprintf('%-8s | %-12s | %-12s | %-12s\n', ...
    'maxit', 'SDR_bsl(dB)', 'SDR_lsmr(dB)', 'diff(dB)');
fprintf('%s\n', repmat('-', 1, 52));

for maxit = maxit_values
    % Backslash reference
    r_bsl = ainmf_modified('AM', y, mask, K, maxit, ...
        'M', M, 'a', a, 'F', F, 'nmfit', nmfit, ...
        'solver', 'backslash', 'verbose', false, 'saveall', false);
    sdr_bsl = 20*log10(norm(y(gap_idx)) / norm(y(gap_idx) - r_bsl(gap_idx)));

    % lsmr_op
    r_lop = ainmf_modified('AM', y, mask, K, maxit, ...
        'M', M, 'a', a, 'F', F, 'nmfit', nmfit, ...
        'solver', 'lsmr_op', ...
        'lsmr_atol', lsmr_atol, 'lsmr_itnlim', lsmr_itnlim, ...
        'verbose', false, 'saveall', false);
    sdr_lop = 20*log10(norm(y(gap_idx)) / norm(y(gap_idx) - r_lop(gap_idx)));

    fprintf('%-8d | %-12.4f | %-12.4f | %-12.4f\n', ...
        maxit, sdr_bsl, sdr_lop, sdr_lop - sdr_bsl);
end

% -------------------------------------------------------------------------
%  Sub-experiment B: inline compare mode — per-AM-iteration error report
%  solver='compare' computes all four solvers per frame and prints:
%    mean/max of ||z_lsmr_op - z_bsl|| / ||z_bsl||  (solve error)
%    mean/max of ||s_lsmr_op - s_bsl|| / ||s_bsl||  (spectral coeff error)
%    mean/max LSMR iteration count, number of itnlim hits
% -------------------------------------------------------------------------
fprintf('\n=== Sub-experiment B: inline per-frame compare (maxit=20) ===\n');
fprintf('Each line = one AM iteration.\n');
fprintf('err_z = ||z_lsmr_op - z_bsl|| / ||z_bsl||  (solve error)\n');
fprintf('err_s = ||s_lsmr_op - s_bsl|| / ||s_bsl||  (spectral coeff error)\n\n');

r_cmp = ainmf_modified('AM', y, mask, K, 20, ...
    'M', M, 'a', a, 'F', F, 'nmfit', nmfit, ...
    'solver', 'compare', ...
    'lsmr_atol', lsmr_atol, 'lsmr_itnlim', lsmr_itnlim, ...
    'verbose', false, 'saveall', false);

sdr_cmp = 20*log10(norm(y(gap_idx)) / norm(y(gap_idx) - r_cmp(gap_idx)));
fprintf('\nFinal SDR (compare mode, uses backslash result): %.4f dB\n', sdr_cmp);