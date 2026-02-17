%% Comparison of Convergence Speed: F.4 vs F.5

clear; clc; close all;

% Problem setup
N = 100; M = 0.6 * N;
rng(1); % Reproducibility
T = randn(M, N);
x_gt = randn(N, 1);
y = T * x_gt;
x_sol = pinv(T) * y;

% Spectral bounds
R = T' * T;
eigs = eig(R);
B = max(eigs);
A = min(eigs(eigs > 1e-10));

% Optimal lambda selection
lam_F4 = 2 / (A + B);          % Optimal for F.4
lam_F5 = sqrt(2 / (A^2 + B^2)); % Optimal for F.5

% Pre-calculate contraction factors
r_F4 = (B - A) / (B + A);
r_F5 = (B^2 - A^2) / (B^2 + A^2);

max_iter = 1000;
errors_F4 = zeros(max_iter, 1);
errors_F5 = zeros(max_iter, 1);

% --- Algorithm F.4 ---
x1_F4 = lam_F4 * T' * y;
xk = x1_F4;
for n = 1:max_iter
    errors_F4(n) = norm(xk - x_sol);
    xk = xk + x1_F4 - lam_F4 * R * xk; % [cite: 24]
end

% --- Algorithm F.5 ---
x1_F5 = (lam_F5^2) * R * (T' * y);
xk = x1_F5;
for n = 1:max_iter
    errors_F5(n) = norm(xk - x_sol);
    xk = xk + x1_F5 - (lam_F5^2) * (R * R * xk); % [cite: 39]
end

% Plotting the Comparison
figure('Position', [100, 100, 800, 600]);
semilogy(errors_F4, 'b', 'LineWidth', 2, 'DisplayName', 'F.4 (Linear Operator R)');
hold on;
semilogy(errors_F5, 'r--', 'LineWidth', 2, 'DisplayName', 'F.5 (Quadratic Operator R^2)');
grid on;
title('Convergence Speed Comparison: F.4 vs F.5');
subtitle(sprintf('Contraction Factors: r_{F4} = %.4f, r_{F5} = %.4f', r_F4, r_F5));
xlabel('Iteration');
ylabel('Error ||x_n - x_{sol}||');
legend('Location', 'northeast');

fprintf('Theoretical Contraction Factors:\n');
fprintf('  F.4 (Linear): r = %.6f\n', r_F4);
fprintf('  F.5 (Quadratic): r = %.6f\n', r_F5);
