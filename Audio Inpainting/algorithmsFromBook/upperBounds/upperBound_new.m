%% Systematic Convergence Analysis for Theorems F.4 and F.5
% Measures iterations required to reach error tolerance of 10^-9

clear; clc; close all;

% 1. Define Experimental Parameters
target_tol = 1e-9;
max_iter = 10000; % High limit for poorly conditioned cases
test_cases = [
    500,  100;   % Case 1: Very skinny (low kappa)
    500,  300;   % Case 2: Skinny
    1000, 600;   % Case 3: Supervisor's small shape example
    1000, 900;   % Case 4: Near square (high kappa)
    2000, 1200;  % Case 5: Larger scale
];

results = []; % To store: [M, N, Kappa, Iter_F4, Iter_F5]

fprintf('Starting Systematic Analysis...\n');
fprintf('%-10s %-10s %-10s %-12s %-12s\n', 'Size M', 'Size N', 'Kappa', 'Iter F.4', 'Iter F.5');
fprintf('----------------------------------------------------------------------\n');

for i = 1:size(test_cases, 1)
    M = test_cases(i, 1);
    N = test_cases(i, 2);
    
    % Generate Matrix
    rng(i); % Consistent randoms
    T = randn(M, N);
    x_gt = randn(N, 1);
    y = T * x_gt;
    x_sol = pinv(T) * y;
    
    % Spectral Bounds
    R = T' * T;
    eigs = eig(R);
    B = max(eigs);
    A = min(eigs(eigs > 1e-10));
    kappa = sqrt(B/A);
    
    % Optimal Lambdas
    lam_f2 = 2 / (A + B);
    lam_f3 = sqrt(2 / (A^2 + B^2));
    
    % --- Test F.4 (Theorem F.2) ---
    x_f2 = lam_f2 * T' * y;
    x_init_f2 = x_f2;
    iter_f2 = 0;
    curr_err = norm(x_f2 - x_sol);
    while curr_err > target_tol && iter_f2 < max_iter
        iter_f2 = iter_f2 + 1;
        x_f2 = x_f2 + x_init_f2 - lam_f2 * R * x_f2;
        curr_err = norm(x_f2 - x_sol);
    end
    
    % --- Test F.5 (Theorem F.3) ---
    x_init_f3 = (lam_f3^2) * R * (T' * y);
    x_f3 = x_init_f3;
    iter_f3 = 0;
    curr_err = norm(x_f3 - x_sol);
    while curr_err > target_tol && iter_f3 < max_iter
        iter_f3 = iter_f3 + 1;
        x_f3 = x_f3 + x_init_f3 - (lam_f3^2) * (R * (R * x_f3));
        curr_err = norm(x_f3 - x_sol);
    end
    
    % Store Results
    results = [results; M, N, kappa, iter_f2, iter_f3];
    fprintf('%-10d %-10d %-10.2f %-12d %-12d\n', M, N, kappa, iter_f2, iter_f3);
end

% 2. Visualization
figure('Position', [100, 100, 800, 500]);
plot(results(:, 3), results(:, 4), 'b-o', 'LineWidth', 2, 'DisplayName', 'F.4 (Linear)');
hold on;
plot(results(:, 3), results(:, 5), 'r--s', 'LineWidth', 2, 'DisplayName', 'F.5 (Quadratic)');
grid on;
title(['Convergence Speed vs Condition Number (\kappa)'], 'FontSize', 14);
subtitle(['Target Tolerance: ' num2str(target_tol)]);
xlabel('Condition Number (\kappa)');
ylabel('Iterations to Converge');
legend('Location', 'northwest');
