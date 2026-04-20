%% Results Section 1 — Convergence Behaviour: Theoretical vs. Observed
%
% Produces two publication-quality figures:
%
%   Figure 1  —  Speed comparison: LSMR vs F.4 vs F.5 on the same axes,
%                for three representative condition numbers (kappa = 2, 5, 10).
%                Shows the practical speed gap.
%
%   Figure 2  —  Bounds validation: observed error vs. theoretical upper
%                bound for F.4 and F.5, at kappa = 2 and 3.
%                (Higher kappa is omitted because F.5 does not converge
%                within a reasonable iteration budget above kappa ~ 3-4.)
%
% Dependencies: generate_matrix.m, f2.m, f3.m, mod_f2.m, mod_f3.m,
%               mod_lsmr.m  (all must be on the MATLAB path)
%
% Notes on condition number choice:
%   - For F.5 (R^2 formulation), the contraction ratio is
%       r_F5 = (B^2 - A^2) / (B^2 + A^2)  (A, B = min/max eigenvalues of T'T)
%     which grows faster with kappa than r_F4 = (B-A)/(B+A),
%     making F.5 empirically slower and unable to converge for kappa >= ~5
%     within 5000 iterations at the tested matrix sizes.
%   - LSMR uses short-recurrence bidiagonalisation and is dramatically
%     faster than both F.4 and F.5 across all tested condition numbers.

clear; clc; close all;

%% ── Global settings ──────────────────────────────────────────────────────

rng(42);           % reproducibility

M = 1500;          % matrix rows
N = 1000;          % matrix cols  (tall system, M > N)
tol = 1e-9;        % convergence tolerance used by all three algorithms

% Colour palette (blue / green / red-orange / grey)
col_f4   = [0.00, 0.45, 0.74];
col_f5   = [0.47, 0.67, 0.19];
col_lsmr = [0.85, 0.33, 0.10];
col_bnd  = [0.55, 0.55, 0.55];   % grey for theoretical bounds

lw_obs = 1.8;   % line width, observed curves
lw_bnd = 1.4;   % line width, theoretical bound

%% ═══════════════════════════════════════════════════════════════════════
%%  FIGURE 1 — Speed comparison across three condition numbers
%% ═══════════════════════════════════════════════════════════════════════

kappas_speed = [10, 15, 20];
max_iter_speed = 5000;          % generous limit; algorithms stop early via tol

fig1 = figure('Position', [50, 500, 1350, 420]);
tl1  = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

for c = 1:3
    kappa = kappas_speed(c);
    fprintf('[Fig 1] kappa = %d  (%d x %d) ...\n', kappa, M, N);

    % Build matrix and ground-truth solution
    T     = generate_matrix(M, N, kappa);
    x_true = randn(N, 1);
    y     = T * x_true;
    x_gt  = pinv(T) * y;

    % Spectral bounds via SVD (more reliable than eig for large matrices)
    B = svds(T, 1, 'largest')^2;
    A = svds(T, 1, 'smallestnz')^2;

    % Optimal step-size parameters
    lam_f4 = 2 / (A + B);
    lam_f5 = sqrt(2 / (A^2 + B^2));

    % Run all three
    [errors_f4,   ~] = mod_f2(T, y, x_gt, lam_f4, max_iter_speed, tol);
    [errors_f5,   ~] = mod_f3(T, y, x_gt, lam_f5, max_iter_speed, tol);
    [~, errors_lsmr, ~, ~] = mod_lsmr(T, y, 0, tol, tol, 1e8, ...
                                       max_iter_speed, 0, false, x_gt);

    n_f4   = length(errors_f4);
    n_f5   = length(errors_f5);
    n_lsmr = length(errors_lsmr);

    fprintf('         LSMR: %d  |  F.4: %d  |  F.5: %d\n', ...
            n_lsmr, n_f4, n_f5);

    % ── Plot ──
    nexttile(tl1, c);
    semilogy(1:n_lsmr, errors_lsmr, 'Color', col_lsmr, ...
             'LineWidth', lw_obs, 'DisplayName', 'LSMR');
    hold on;
    semilogy(1:n_f4, errors_f4, 'Color', col_f4, ...
             'LineWidth', lw_obs, 'DisplayName', 'Corollary F.4 (Landweber)');
    semilogy(1:n_f5, errors_f5, 'Color', col_f5, ...
             'LineWidth', lw_obs, 'DisplayName', 'Corollary F.5 (R^2)');
    grid on;
    xlabel('Iteration k', 'FontSize', 11);
    ylabel('\|x_k - x^\dagger\|_2', 'FontSize', 11);
    title(sprintf('\\kappa = %d', kappa), 'FontSize', 12);
    if c == 1
        legend('Location', 'southwest', 'FontSize', 9);
    end
    % Mark the tolerance line for reference
    yline(tol, ':', 'Color', [0.4 0.4 0.4], 'LineWidth', 1, ...
          'DisplayName', 'tolerance');
    hold off;
end

title(tl1, sprintf('Convergence speed comparison  —  %d \\times %d matrices', M, N), ...
      'FontSize', 13, 'FontWeight', 'bold');

%% ═══════════════════════════════════════════════════════════════════════
%%  FIGURE 2 — Theoretical upper bound vs. observed error
%% ═══════════════════════════════════════════════════════════════════════
%
%  Layout:   rows    = algorithm  (F.4 top,  F.5 bottom)
%            columns = kappa      (kappa=2 left, kappa=3 right)
%
%  For F.5 at kappa=3 the contraction ratio r_F5 ~ 0.975, requiring
%  ~950 iterations to reach 1e-9 — hence max_iter_bounds = 1200.

kappas_bounds = [6, 15];
max_iter_bounds = 1200;

fig2 = figure('Position', [50, 50, 1050, 600]);
tl2  = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

for c = 1:2
    kappa = kappas_bounds(c);
    fprintf('[Fig 2] kappa = %d  (%d x %d) ...\n', kappa, M, N);

    % Build matrix and ground-truth solution
    T     = generate_matrix(M, N, kappa);
    x_true = randn(N, 1);
    y     = T * x_true;
    x_gt  = pinv(T) * y;

    % Spectral bounds
    B = svds(T, 1, 'largest')^2;
    A = svds(T, 1, 'smallestnz')^2;

    % Optimal parameters
    lam_f4 = 2 / (A + B);
    lam_f5 = sqrt(2 / (A^2 + B^2));

    % Contraction ratios (for the upper bound formulae)
    r_f4 = (B - A) / (B + A);
    r_f5 = (B^2 - A^2) / (B^2 + A^2);

    % Theoretical upper bounds: ||x_k - x†|| <= r^k * ||x†||
    %   (evaluated at iterations 1, 2, ..., max_iter_bounds)
    x_gt_norm  = norm(x_gt);
    k_range    = (1:max_iter_bounds)';
    bound_f4   = (r_f4 .^ k_range) * x_gt_norm;
    bound_f5   = (r_f5 .^ k_range) * x_gt_norm;

    % Run WITHOUT early stopping to see the full curve
    [errors_f4_full, ~] = f2(T, y, x_gt, lam_f4, max_iter_bounds);
    [errors_f5_full, ~] = f3(T, y, x_gt, lam_f5, max_iter_bounds);

    fprintf('         r_F4 = %.4f  |  r_F5 = %.4f\n', r_f4, r_f5);

    % ── Top row: F.4 ──
    nexttile(tl2, c);          % tile 1 (kappa=2) or tile 2 (kappa=3)
    semilogy(1:max_iter_bounds, errors_f4_full, ...
             'Color', col_f4, 'LineWidth', lw_obs, ...
             'DisplayName', 'Observed error');
    hold on;
    semilogy(k_range, bound_f4, '--', ...
             'Color', col_bnd, 'LineWidth', lw_bnd, ...
             'DisplayName', 'Theoretical upper bound');
    grid on;
    xlabel('Iteration k', 'FontSize', 11);
    ylabel('\|x_k - x^\dagger\|_2', 'FontSize', 11);
    title(sprintf('Corollary F.4  —  \\kappa = %d', kappa), 'FontSize', 11);
    if c == 1
        legend('Location', 'northeast', 'FontSize', 9);
    end
    hold off;

    % ── Bottom row: F.5 ──
    nexttile(tl2, 2 + c);     % tile 3 or tile 4
    semilogy(1:max_iter_bounds, errors_f5_full, ...
             'Color', col_f5, 'LineWidth', lw_obs, ...
             'DisplayName', 'Observed error');
    hold on;
    semilogy(k_range, bound_f5, '--', ...
             'Color', col_bnd, 'LineWidth', lw_bnd, ...
             'DisplayName', 'Theoretical upper bound');
    grid on;
    xlabel('Iteration k', 'FontSize', 11);
    ylabel('\|x_k - x^\dagger\|_2', 'FontSize', 11);
    title(sprintf('Corollary F.5  —  \\kappa = %d', kappa), 'FontSize', 11);
    if c == 1
        legend('Location', 'northeast', 'FontSize', 9);
    end
    hold off;
end

title(tl2, sprintf('Observed error vs. theoretical upper bound  —  %d \\times %d matrices', M, N), ...
      'FontSize', 13, 'FontWeight', 'bold');

%% ── Export ───────────────────────────────────────────────────────────────

saveas(fig1, 'fig1_convergence_speed.png');
saveas(fig2, 'fig2_convergence_bounds.png');
fprintf('\nFigures saved: fig1_convergence_speed.png  |  fig2_convergence_bounds.png\n');
