%% plots.m
%
% Plots produced:
%   1. itnlim vs SDR       (M=1024, both solvers)
%   2. itnlim vs Time      (M=1024, both solvers)
%   3. Time vs SDR         (M=1024, both solvers — Pareto frontier)
%   4. atol vs SDR         (M=1024, lsmr_op, itnlim=500)
%   5. atol vs Time        (M=1024, lsmr_op, itnlim=500)
%   6. SDR vs M            (scaling across M=1024,2048,4096)
%   7. Time vs M           (scaling across M=1024,2048,4096)

clear; clc; close all;

% =========================================================================
%  Data from benchmarks
%  All runs: AM method, mamavatu.wav at 16kHz, 50% random gap,
%            K=10, maxit=20, nmfit=1, atol=0 (unless noted)
% =========================================================================

%% --- M=1024, F=2048 itnlim sweep ----------------------------------------
itnlim_vals = [500, 1000, 2000, 5000];

time_mat  = [278,  529,  641,  787];
sdr_mat   = [14.06, 14.57, 15.43, 15.65];

time_op   = [352,  309,  393,  433];
sdr_op    = [14.09, 14.57, 15.43, 15.67];

% Reference lines (exact solvers, M=1024)
time_bsl_1024  = 218;
sdr_bsl_1024   = 14.76;
time_chol_1024 = 219;
sdr_chol_1024  = 14.76;

%% --- atol sweep (M=1024, lsmr_op, itnlim=500) ---------------------------
% atol: 1e-6, 1e-9, 0
atol_vals    = [1e-6,  1e-9,  0];
atol_labels  = {'1e-6', '1e-9', '0'};
time_atol_op = [377,   446,   272];
sdr_atol_op  = [13.18, 14.09, 14.09];

%% --- Scaling across M ---------------------------------------------------
M_vals = [1024, 2048, 4096];

% Exact solvers (backslash — cleaner timing at each M)
time_bsl   = [218,  787,  1868];
sdr_bsl    = [14.76, 16.14, 16.49];

% lsmr_op at itnlim=1000
time_op_1k = [309,  1010, 1636];
sdr_op_1k  = [14.57, 14.87, 15.47];

% lsmr_op at itnlim=2000
time_op_2k = [393,  1186, 2604];
sdr_op_2k  = [15.43, 15.56, 16.27];


% =========================================================================
%  Plot 1 — itnlim vs SDR (M=1024)
% =========================================================================
figure('Name', 'itnlim vs SDR');
plot(itnlim_vals, sdr_mat, 'b-o', 'LineWidth', 1.8, 'MarkerSize', 7, ...
    'DisplayName', 'lsmr\_matrix');
hold on;
plot(itnlim_vals, sdr_op, 'r-s', 'LineWidth', 1.8, 'MarkerSize', 7, ...
    'DisplayName', 'lsmr\_op');
yline(sdr_bsl_1024, 'k--', 'LineWidth', 1.4, 'DisplayName', 'backslash / cholesky');
grid on;
xlabel('itnlim');
ylabel('SDR (dB)');
title('Iteration limit vs reconstruction quality  (M=1024)');
legend('Location', 'southeast');
xlim([0 5500]);
saveas(gcf, 'plot1_itnlim_vs_sdr.png');

% =========================================================================
%  Plot 2 — itnlim vs Time (M=1024)
% =========================================================================
figure('Name', 'itnlim vs Time');
plot(itnlim_vals, time_mat, 'b-o', 'LineWidth', 1.8, 'MarkerSize', 7, ...
    'DisplayName', 'lsmr\_matrix');
hold on;
plot(itnlim_vals, time_op, 'r-s', 'LineWidth', 1.8, 'MarkerSize', 7, ...
    'DisplayName', 'lsmr\_op');
yline(time_bsl_1024,  'k--', 'LineWidth', 1.4, 'DisplayName', 'backslash');
yline(time_chol_1024, 'k:',  'LineWidth', 1.4, 'DisplayName', 'cholesky');
grid on;
xlabel('itnlim');
ylabel('Wall-clock time (s)');
title('Iteration limit vs computation time  (M=1024)');
legend('Location', 'northwest');
xlim([0 5500]);
saveas(gcf, 'plot2_itnlim_vs_time.png');

% =========================================================================
%  Plot 3 — Time vs SDR  (Pareto frontier, M=1024)
% =========================================================================
figure('Name', 'Time vs SDR');
plot(time_mat, sdr_mat, 'b-o', 'LineWidth', 1.8, 'MarkerSize', 7, ...
    'DisplayName', 'lsmr\_matrix');
hold on;
plot(time_op, sdr_op, 'r-s', 'LineWidth', 1.8, 'MarkerSize', 7, ...
    'DisplayName', 'lsmr\_op');
% Label itnlim on each point
for k = 1:length(itnlim_vals)
    text(time_mat(k)+8, sdr_mat(k), sprintf('%d', itnlim_vals(k)), ...
        'FontSize', 9, 'Color', 'b');
    text(time_op(k)+8,  sdr_op(k),  sprintf('%d', itnlim_vals(k)), ...
        'FontSize', 9, 'Color', 'r');
end
% Exact solver reference point
plot(time_bsl_1024, sdr_bsl_1024, 'k*', 'MarkerSize', 12, ...
    'LineWidth', 2, 'DisplayName', 'backslash / cholesky');
grid on;
xlabel('Wall-clock time (s)');
ylabel('SDR (dB)');
title('Accuracy--speed tradeoff  (M=1024)');
legend('Location', 'southeast');
saveas(gcf, 'plot3_time_vs_sdr.png');

% =========================================================================
%  Plot 4 — atol vs SDR (M=1024, lsmr_op, itnlim=500)
% =========================================================================
figure('Name', 'atol vs SDR');
semilogx([1e-6, 1e-9, 1e-15], sdr_atol_op, 'r-s', ...
    'LineWidth', 1.8, 'MarkerSize', 8);
hold on;
yline(sdr_bsl_1024, 'k--', 'LineWidth', 1.4, 'DisplayName', 'backslash');
grid on;
xlabel('atol');
ylabel('SDR (dB)');
title('Tolerance vs reconstruction quality  (lsmr\_op, M=1024, itnlim=500)');
% Use 0 as ~machine epsilon on the log axis
xticks([1e-15 1e-9 1e-6]);
xticklabels({'0 (eps)', '1e-9', '1e-6'});
legend({'lsmr\_op', 'backslash / cholesky'}, 'Location', 'southeast');
annotation('textbox', [0.15 0.2 0.5 0.1], 'String', ...
    'atol has no measurable effect on SDR — itnlim is the binding constraint', ...
    'EdgeColor', 'none', 'FontSize', 9, 'Color', [0.4 0.4 0.4]);
saveas(gcf, 'plot4_atol_vs_sdr.png');

% =========================================================================
%  Plot 5 — atol vs Time (M=1024, lsmr_op, itnlim=500)
% =========================================================================
figure('Name', 'atol vs Time');
semilogx([1e-6, 1e-9, 1e-15], time_atol_op, 'r-s', ...
    'LineWidth', 1.8, 'MarkerSize', 8);
hold on;
yline(time_chol_1024, 'k--', 'LineWidth', 1.4);
grid on;
xlabel('atol');
ylabel('Wall-clock time (s)');
title('Tolerance vs computation time  (lsmr\_op, M=1024, itnlim=500)');
xticks([1e-15 1e-9 1e-6]);
xticklabels({'0 (eps)', '1e-9', '1e-6'});
legend({'lsmr\_op', 'cholesky'}, 'Location', 'northeast');
saveas(gcf, 'plot5_atol_vs_time.png');

% =========================================================================
%  Plot 6 — SDR vs M  (scaling)
% =========================================================================
figure('Name', 'SDR vs M');
plot(M_vals, sdr_bsl,   'k-*',  'LineWidth', 1.8, 'MarkerSize', 9, ...
    'DisplayName', 'backslash (exact)');
hold on;
plot(M_vals, sdr_op_1k, 'r--s', 'LineWidth', 1.8, 'MarkerSize', 7, ...
    'DisplayName', 'lsmr\_op  itnlim=1000');
plot(M_vals, sdr_op_2k, 'b--o', 'LineWidth', 1.8, 'MarkerSize', 7, ...
    'DisplayName', 'lsmr\_op  itnlim=2000');
grid on;
xlabel('Window length M');
ylabel('SDR (dB)');
title('Reconstruction quality vs problem scale');
legend('Location', 'southeast');
xticks(M_vals);
xticklabels({'1024', '2048', '4096'});
saveas(gcf, 'plot6_sdr_vs_M.png');

% =========================================================================
%  Plot 7 — Time vs M  (scaling)
% =========================================================================
figure('Name', 'Time vs M');
plot(M_vals, time_bsl,   'k-*',  'LineWidth', 1.8, 'MarkerSize', 9, ...
    'DisplayName', 'backslash (exact)');
hold on;
plot(M_vals, time_op_1k, 'r--s', 'LineWidth', 1.8, 'MarkerSize', 7, ...
    'DisplayName', 'lsmr\_op  itnlim=1000');
plot(M_vals, time_op_2k, 'b--o', 'LineWidth', 1.8, 'MarkerSize', 7, ...
    'DisplayName', 'lsmr\_op  itnlim=2000');
grid on;
xlabel('Window length M');
ylabel('Wall-clock time (s)');
title('Computation time vs problem scale');
legend('Location', 'northwest');
xticks(M_vals);
xticklabels({'1024', '2048', '4096'});
saveas(gcf, 'plot7_time_vs_M.png');

fprintf('\nAll 7 plots saved as PNG files.\n');
