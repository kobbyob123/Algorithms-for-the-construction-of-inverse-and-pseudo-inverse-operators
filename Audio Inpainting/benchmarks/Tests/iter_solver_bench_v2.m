clc; clear; close all;

% Solving for under-determined systems
M = 1000;
N = 600;
kappa = 10;
tol = 1e-9;
itnlim = 2000;

T_test = generate_matrix(M, N, kappa);
x_original = randn(N, 1);
y_test = T_test * x_original;

x_gt = pinv(T_test) * y_test; 

R_test = T_test' * T_test;
eigs_test = eig(R_test);
B_test = max(eigs_test);
A_test = min(eigs_test(eigs_test > 1e-10));
lambda_F2 = 2 / (A_test + B_test);
lambda_F3 = sqrt(2 / (A_test^2 + B_test^2));

[~, errors_lsmr, ~, itn_lsmr] = mod_lsmr(T_test, y_test, 0, tol, tol, 1e12, itnlim, 0, false, x_gt);
[errors_F2, ~] = mod_f2(T_test, y_test, x_gt, lambda_F2, itnlim, tol);
[errors_F3, ~] = mod_f3(T_test, y_test, x_gt, lambda_F3, itnlim, tol);

% Showing some output
fprintf('Iterations to reach %g tolerance (m < n):\n', tol);
fprintf('LSMR: %d iterations\n', length(errors_lsmr));
fprintf('F2:   %d iterations\n', length(errors_F2));
fprintf('F3:   %d iterations\n', length(errors_F3));

% Showing some graphs

figure;
% Plot F2 in blue
semilogy(1:length(errors_F2), errors_F2, 'b', 'LineWidth', 1.5, 'DisplayName', 'F2 (Corollary F.4)');
hold on;

% Plot F3 in green
semilogy(1:length(errors_F3), errors_F3, 'g', 'LineWidth', 1.5, 'DisplayName', 'F3 (Corollary F.5)');

% Plot LSMR in red
semilogy(1:length(errors_lsmr), errors_lsmr, 'r', 'LineWidth', 1.5, 'DisplayName', 'LSMR');

% Formatting the plot
grid on;
title('Convergence Comparison: Iterative Inverse Algorithms');
xlabel('Iteration count (k)');
ylabel('Distance to Ground Truth: ||x_k - x_{gt}||_2');
legend('show');
hold off;
