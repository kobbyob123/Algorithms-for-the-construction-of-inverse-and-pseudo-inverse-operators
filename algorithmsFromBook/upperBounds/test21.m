%% Convergence Analysis: Theorems F.2 vs F.3
%
% Supervisor request: systematic study of convergence speed as a function
% of matrix shape (M/N ratio) and condition number κ(T).
%
% Two convergence metrics:
%   (1) iter_to_tol  – number of iterations to reach relative error < 1e-9
%   (2) log_rate     – empirical log-decrease per iteration
%                      (slope of log10(error) vs. iteration, fitted by OLS)
%
% Theoretical prediction (optimal λ):
%   r_F2 = (κ²-1)/(κ²+1),   log_rate_F2 = log10(r_F2)
%   r_F3 = (κ⁴-1)/(κ⁴+1),   log_rate_F3 = log10(r_F3)
%   iters_to_tol ≈ log(tol) / log(r)        (from ||x_n - x_0|| ≤ r^n ||x_0||)
%
% Matrix construction: T = U * Σ * V'  with prescribed singular values,
% so κ(T) = σ_max / σ_min is exact.  Shape (M×N) controls the rank
% (min(M,N) nonzero singular values).

clear; clc; close all;
rng(42);

%% ── Experiment Parameters ────────────────────────────────────────────────

N        = 300;                       % base column dimension (fixed)
MN_ratios = [0.10, 0.25, 0.50, 0.75, 0.90];  % M/N shapes to test
kappa_vals = [2, 4, 9, 20, 50];       % condition numbers κ(T) to test

tol      = 1e-9;                      % relative-error target
max_iter = 5000;                      % hard cap on iterations

n_shapes  = numel(MN_ratios);
n_kappas  = numel(kappa_vals);

% Storage: rows = κ values, cols = M/N shapes
iters_F2   = NaN(n_kappas, n_shapes);
iters_F3   = NaN(n_kappas, n_shapes);
rate_F2    = NaN(n_kappas, n_shapes);
rate_F3    = NaN(n_kappas, n_shapes);
r_theo_F2  = NaN(n_kappas, n_shapes);
r_theo_F3  = NaN(n_kappas, n_shapes);

%% ── Helper: build T with exact κ ─────────────────────────────────────────
function T = make_T(M, N, kappa)
    % Returns an M×N matrix with κ(T) = kappa.
    % Singular values are log-spaced between 1 and kappa.
    r  = min(M, N);
    sv = logspace(0, log10(kappa), r);   % σ_min=1, σ_max=kappa
    [U, ~] = qr(randn(M, M));
    [V, ~] = qr(randn(N, N));
    S = zeros(M, N);
    for i = 1:r, S(i,i) = sv(i); end
    T = U * S * V';
end

%% ── Helper: empirical log-convergence rate ────────────────────────────────
function rate = fit_log_rate(errors, tol_abs)
    % Fit log10(error) ~ a + rate*iter on the portion above tol_abs.
    % 'rate' is the per-iteration log10-decrease (negative → converging).
    valid = errors > tol_abs & errors < errors(1)*1.1;  % remove tail noise
    if sum(valid) < 5
        rate = NaN; return;
    end
    idx = find(valid);
    p   = polyfit(idx, log10(errors(idx)), 1);
    rate = p(1);   % slope (negative = converging)
end

%% ── Main experiment loop ─────────────────────────────────────────────────
fprintf('Running experiments  (N=%d, %d shapes × %d κ values)\n\n', ...
        N, n_shapes, n_kappas);

for ki = 1:n_kappas
    kap = kappa_vals(ki);
    B_over_A = kap^2;                  % B/A = κ(T)² = κ(R)

    for si = 1:n_shapes
        M = max(1, round(MN_ratios(si) * N));

        % --- build matrix & problem ---
        T   = make_T(M, N, kap);
        x_gt = randn(N, 1);
        y    = T * x_gt;
        x_sol = pinv(T) * y;           % minimum-norm least-squares solution

        % --- spectral bounds from R = T'T ---
        R     = T' * T;
        ev    = sort(eig(R), 'ascend');
        B_sp  = ev(end);
        A_sp  = min(ev(ev > 1e-12));

        % Theoretical convergence factors
        r2 = (B_sp - A_sp) / (B_sp + A_sp);
        r3 = (B_sp^2 - A_sp^2) / (B_sp^2 + A_sp^2);
        r_theo_F2(ki, si) = r2;
        r_theo_F3(ki, si) = r3;

        % Optimal λ values
        lam_f2 = 2 / (A_sp + B_sp);
        lam_f3 = sqrt(2 / (A_sp^2 + B_sp^2));

        % --- run algorithms ---
        [err2, ~] = f2(T, y, x_sol, lam_f2, max_iter);
        [err3, ~] = f3(T, y, x_sol, lam_f3, max_iter);

        % Relative errors
        nrm = norm(x_sol);
        rel2 = err2 / nrm;
        rel3 = err3 / nrm;

        % Metric 1: iterations to tolerance
        idx2 = find(rel2 <= tol, 1, 'first');
        idx3 = find(rel3 <= tol, 1, 'first');
        iters_F2(ki, si) = idx2;   % NaN if never reached
        iters_F3(ki, si) = idx3;

        % Metric 2: empirical log-rate
        tol_abs = tol * nrm;
        rate_F2(ki, si) = fit_log_rate(err2, tol_abs);
        rate_F3(ki, si) = fit_log_rate(err3, tol_abs);

        fprintf('  κ=%4d, M/N=%.2f : iters F2=%5s  F3=%5s | rate F2=%.3f  F3=%.3f\n', ...
            kap, MN_ratios(si), ...
            num2str(idx2, '%5.0f'), num2str(idx3, '%5.0f'), ...
            rate_F2(ki,si), rate_F3(ki,si));
    end
    fprintf('\n');
end

%% ── Figure 1 – Heatmaps: iterations to reach tol ─────────────────────────
shape_labels = arrayfun(@(r) sprintf('%.2f', r), MN_ratios, 'UniformOutput', false);
kappa_labels = arrayfun(@(k) sprintf('%d', k), kappa_vals, 'UniformOutput', false);

figure('Name', 'Iterations to tolerance', 'Position', [50 50 1300 500]);

subplot(1,2,1);
imagesc(iters_F2);
colormap(gca, flipud(hot));
colorbar;
set(gca, 'XTick', 1:n_shapes, 'XTickLabel', shape_labels, ...
         'YTick', 1:n_kappas, 'YTickLabel', kappa_labels, 'FontSize', 11);
xlabel('M/N shape ratio');
ylabel('\kappa(T)');
title(sprintf('F.2 (R): iterations to rel. error < %.0e', tol), 'FontSize', 13);
for ki=1:n_kappas
    for si=1:n_shapes
        v = iters_F2(ki,si);
        if isnan(v), txt = '>max'; else, txt = sprintf('%d', v); end
        text(si, ki, txt, 'HorizontalAlignment','center', ...
             'Color','w', 'FontSize', 9, 'FontWeight','bold');
    end
end

subplot(1,2,2);
imagesc(iters_F3);
colormap(gca, flipud(hot));
colorbar;
set(gca, 'XTick', 1:n_shapes, 'XTickLabel', shape_labels, ...
         'YTick', 1:n_kappas, 'YTickLabel', kappa_labels, 'FontSize', 11);
xlabel('M/N shape ratio');
ylabel('\kappa(T)');
title(sprintf('F.3 (R²): iterations to rel. error < %.0e', tol), 'FontSize', 13);
for ki=1:n_kappas
    for si=1:n_shapes
        v = iters_F3(ki,si);
        if isnan(v), txt = '>max'; else, txt = sprintf('%d', v); end
        text(si, ki, txt, 'HorizontalAlignment','center', ...
             'Color','w', 'FontSize', 9, 'FontWeight','bold');
    end
end

sgtitle('Iterations to convergence  (target: 10^{-9} relative error)', 'FontSize', 15);

%% ── Figure 2 – Heatmaps: empirical log-rate ──────────────────────────────
figure('Name', 'Empirical log-convergence rate', 'Position', [50 600 1300 500]);

subplot(1,2,1);
imagesc(rate_F2);
colormap(gca, hot);          % more negative (darker) = faster convergence
colorbar;
set(gca, 'XTick', 1:n_shapes, 'XTickLabel', shape_labels, ...
         'YTick', 1:n_kappas, 'YTickLabel', kappa_labels, 'FontSize', 11);
xlabel('M/N shape ratio');
ylabel('\kappa(T)');
title('F.2: log_{10}-decrease per iteration', 'FontSize', 13);
for ki=1:n_kappas
    for si=1:n_shapes
        text(si, ki, sprintf('%.3f', rate_F2(ki,si)), ...
             'HorizontalAlignment','center','Color','w','FontSize',9,'FontWeight','bold');
    end
end

subplot(1,2,2);
imagesc(rate_F3);
colormap(gca, hot);
colorbar;
set(gca, 'XTick', 1:n_shapes, 'XTickLabel', shape_labels, ...
         'YTick', 1:n_kappas, 'YTickLabel', kappa_labels, 'FontSize', 11);
xlabel('M/N shape ratio');
ylabel('\kappa(T)');
title('F.3: log_{10}-decrease per iteration', 'FontSize', 13);
for ki=1:n_kappas
    for si=1:n_shapes
        text(si, ki, sprintf('%.3f', rate_F3(ki,si)), ...
             'HorizontalAlignment','center','Color','w','FontSize',9,'FontWeight','bold');
    end
end

sgtitle('Empirical convergence rate  (log_{10} error drop per iteration)', 'FontSize', 15);

%% ── Figure 3 – Theoretical rate vs κ  (shape-independent prediction) ─────
figure('Name', 'Theoretical rate vs kappa', 'Position', [700 50 750 500]);

kap_fine = linspace(1.01, 55, 500);
r2_theo = (kap_fine.^2 - 1) ./ (kap_fine.^2 + 1);
r3_theo = (kap_fine.^4 - 1) ./ (kap_fine.^4 + 1);

semilogy(kap_fine, 1-r2_theo, 'b-',  'LineWidth', 2.5, 'DisplayName', 'F.2  per-iter. reduction (1-r)');
hold on;
semilogy(kap_fine, 1-r3_theo, 'r--', 'LineWidth', 2.5, 'DisplayName', 'F.3  per-iter. reduction (1-r)');
grid on;
xlabel('\kappa(T)  (condition number)', 'FontSize', 13);
ylabel('Per-iteration error reduction  1 - r_\lambda', 'FontSize', 13);
title('Theoretical convergence rate  vs.  \kappa(T)', 'FontSize', 14);
legend('Location', 'northeast', 'FontSize', 12);

% Annotate supervisor's examples
for kap_ex = [2, 9]
    r2_ex = (kap_ex^2-1)/(kap_ex^2+1);
    r3_ex = (kap_ex^4-1)/(kap_ex^4+1);
    xline(kap_ex, 'k:', 'LineWidth', 1);
    text(kap_ex+0.5, 0.8, sprintf('\\kappa=%d', kap_ex), 'FontSize', 11);
end

%% ── Figure 4 – Error trajectories for selected (κ, shape) pairs ──────────
sel_kappas = [2, 9, 50];
sel_shape  = 3;    % M/N = 0.50

figure('Name', 'Error trajectories', 'Position', [50 50 1400 420]);
colors_F2 = [0.1 0.4 0.9; 0.0 0.6 0.3; 0.6 0.0 0.8];
colors_F3 = [1.0 0.3 0.1; 0.9 0.7 0.0; 0.3 0.3 0.3];

M_traj = max(1, round(MN_ratios(sel_shape) * N));

for pi_idx = 1:numel(sel_kappas)
    subplot(1, numel(sel_kappas), pi_idx);
    kap = sel_kappas(pi_idx);

    T    = make_T(M_traj, N, kap);
    x_gt = randn(N,1);
    y    = T * x_gt;
    x_sol = pinv(T) * y;

    ev    = sort(eig(T'*T), 'ascend');
    B_sp  = ev(end);  A_sp = min(ev(ev>1e-12));

    lam_f2 = 2/(A_sp+B_sp);
    lam_f3 = sqrt(2/(A_sp^2+B_sp^2));
    r2     = (B_sp-A_sp)/(B_sp+A_sp);
    r3     = (B_sp^2-A_sp^2)/(B_sp^2+A_sp^2);

    n_traj = min(max_iter, 2000);
    [e2,~] = f2(T, y, x_sol, lam_f2, n_traj);
    [e3,~] = f3(T, y, x_sol, lam_f3, n_traj);
    nrm    = norm(x_sol);

    its = (1:n_traj)';
    semilogy(its, e2/nrm, '-',  'Color', colors_F2(pi_idx,:), 'LineWidth', 2, ...
             'DisplayName', 'F.2 (observed)');
    hold on;
    semilogy(its, e3/nrm, '--', 'Color', colors_F3(pi_idx,:), 'LineWidth', 2, ...
             'DisplayName', 'F.3 (observed)');
    semilogy(its, r2.^its, ':', 'Color', colors_F2(pi_idx,:)*0.7, 'LineWidth', 1.2, ...
             'DisplayName', sprintf('F.2 bound (r=%.4f)', r2));
    semilogy(its, r3.^its, '-.','Color', colors_F3(pi_idx,:)*0.7, 'LineWidth', 1.2, ...
             'DisplayName', sprintf('F.3 bound (r=%.4f)', r3));
    yline(tol, 'k--', sprintf('tol=%.0e', tol), 'LineWidth', 1, 'LabelHorizontalAlignment','right');
    grid on;
    legend('Location', 'northeast', 'FontSize', 8);
    title(sprintf('\\kappa(T) = %d,  M/N = %.2f', kap, MN_ratios(sel_shape)), 'FontSize', 13);
    xlabel('Iteration');
    ylabel('Relative error  ||x_n - x_0|| / ||x_0||');
    set(gca, 'FontSize', 10);
    hold off;
end
sgtitle('Error trajectories with theoretical bounds', 'FontSize', 15);

%% ── Figure 5 – Ratio F3/F2 iterations (overhead cost of stability) ───────
figure('Name', 'Ratio F3 vs F2', 'Position', [700 550 750 480]);

ratio_iters = iters_F3 ./ iters_F2;
imagesc(ratio_iters);
colormap(hot);
colorbar;
set(gca, 'XTick', 1:n_shapes, 'XTickLabel', shape_labels, ...
         'YTick', 1:n_kappas, 'YTickLabel', kappa_labels, 'FontSize', 11);
xlabel('M/N shape ratio');
ylabel('\kappa(T)');
title('Iteration overhead  F.3 / F.2  (values > 1 mean F.3 is slower)', 'FontSize', 13);
for ki=1:n_kappas
    for si=1:n_shapes
        v = ratio_iters(ki,si);
        if isnan(v), txt = 'N/A'; else, txt = sprintf('×%.1f', v); end
        text(si, ki, txt, 'HorizontalAlignment','center', ...
             'Color','w','FontSize',10,'FontWeight','bold');
    end
end

%% ── Printed Summary Table ────────────────────────────────────────────────
fprintf('\n══════════════════════════════════════════════════════════════════\n');
fprintf('  CONVERGENCE SUMMARY  (N=%d, tol=%.0e)\n', N, tol);
fprintf('══════════════════════════════════════════════════════════════════\n\n');

fprintf('── Iterations to rel. error < %.0e ─────────────────────────────\n', tol);
fprintf('          ');
for si=1:n_shapes, fprintf('  M/N=%.2f', MN_ratios(si)); end
fprintf('\n');

for ki=1:n_kappas
    fprintf('κ=%3d F.2 ', kappa_vals(ki));
    for si=1:n_shapes
        v = iters_F2(ki,si);
        if isnan(v), fprintf('     >max'); else, fprintf('     %4d', v); end
    end
    fprintf('\n');
    fprintf('      F.3 ');
    for si=1:n_shapes
        v = iters_F3(ki,si);
        if isnan(v), fprintf('     >max'); else, fprintf('     %4d', v); end
    end
    fprintf('\n');
    fprintf('      ratio ');
    for si=1:n_shapes
        r = iters_F3(ki,si)/iters_F2(ki,si);
        if isnan(r), fprintf('      N/A'); else, fprintf('    ×%.2f', r); end
    end
    fprintf('\n\n');
end

fprintf('── Empirical log₁₀-decrease per iteration ───────────────────────\n');
fprintf('          ');
for si=1:n_shapes, fprintf('  M/N=%.2f', MN_ratios(si)); end
fprintf('\n');
for ki=1:n_kappas
    fprintf('κ=%3d F.2 ', kappa_vals(ki));
    for si=1:n_shapes, fprintf('   %+.4f', rate_F2(ki,si)); end
    fprintf('\n');
    fprintf('      F.3 ');
    for si=1:n_shapes, fprintf('   %+.4f', rate_F3(ki,si)); end
    fprintf('\n\n');
end

fprintf('── Theoretical convergence factors r (optimal λ) ─────────────────\n');
fprintf(' κ(T)   |   r_F2    log₁₀(r_F2)  |   r_F3    log₁₀(r_F3)  | overhead\n');
fprintf(' -------+---------------------------+---------------------------+--------\n');
for ki=1:n_kappas
    kap   = kappa_vals(ki);
    r2    = (kap^2-1)/(kap^2+1);
    r3    = (kap^4-1)/(kap^4+1);
    n_F2  = ceil(log(tol)/log(r2));
    n_F3  = ceil(log(tol)/log(r3));
    fprintf(' %5d  |  %.6f   %+8.4f  |  %.6f   %+8.4f  |  ×%.2f\n', ...
        kap, r2, log10(r2), r3, log10(r3), n_F3/n_F2);
end
fprintf('\n');
