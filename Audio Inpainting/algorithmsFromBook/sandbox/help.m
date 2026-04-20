%% ========================================================================
%% ITERATIVE ALGORITHM ANALYSIS SUITE
%% Based on Appendix F: Věta F.2 and F.3 (Landweber Iteration)
%% ========================================================================

clear; clc; close all;

%% ========================================================================
%% TASK 1 & 2: Basic Implementation with Ground Truth
%% ========================================================================

fprintf('=== TASK 1 & 2: Basic Algorithm Implementation ===\n\n');

% Define problem dimensions
N = 50;  % Signal size
M = 30;  % Observation size (M < N means underdetermined)

% Create random masking operator
rng(42); % For reproducibility
T_matrix = randn(M, N);

% Define true signal
x_true = randn(N, 1);

% Create observed signal
y = T_matrix * x_true;

% Compute ground truth using pseudoinverse
x_groundtruth = pinv(T_matrix) * y;

% Compute spectral bounds for lambda selection
R_matrix = T_matrix' * T_matrix;
eigenvalues = eig(R_matrix);
B = max(eigenvalues);  % Upper bound
A = min(eigenvalues(eigenvalues > 1e-10));  % Lower bound (exclude near-zero)

fprintf('Matrix properties:\n');
fprintf('  Size: %d x %d\n', M, N);
fprintf('  Spectral bounds: A = %.4f, B = %.4f\n', A, B);
fprintf('  Condition number: κ(T) ≈ %.4f\n', sqrt(B/A));
fprintf('  Lambda interval: (0, %.4f)\n\n', 2/B);

% Test with optimal lambda
lambda_optimal = 2 / (A + B);
[errors_opt, x_final_opt] = landweber_iteration(T_matrix, y, x_groundtruth, lambda_optimal, 500);

fprintf('Optimal lambda = %.6f\n', lambda_optimal);
fprintf('Final error: %.6e\n', errors_opt(end));
fprintf('Distance from ground truth: %.6e\n\n', norm(x_final_opt - x_groundtruth));

% Plot convergence
figure('Position', [100, 100, 800, 600]);
semilogy(errors_opt, 'LineWidth', 2, 'Color', [0.2, 0.4, 0.8]);
grid on;
title('Convergence Profile: Landweber Iteration (Optimal λ)', 'FontSize', 14);
xlabel('Iteration Number', 'FontSize', 12);
ylabel('Error ||x_k - x_{ground truth}||', 'FontSize', 12);
set(gca, 'FontSize', 11);

%% ========================================================================
%% TASK 3: Lambda Parameter Study
%% ========================================================================

fprintf('=== TASK 3: Lambda Parameter Study ===\n\n');

% Define lambda values to test
lambda_safe_boundary = 2/B;
lambdas = [
    0.5 * lambda_optimal,          % Conservative
    lambda_optimal,                % Optimal
    1.5 * lambda_optimal,          % Aggressive but safe
    0.9 * lambda_safe_boundary,    % Near boundary
    1.1 * lambda_safe_boundary,    % Exceeds boundary (should diverge)
    1.5 * lambda_safe_boundary     % Well beyond (should diverge faster)
];

lambda_labels = {
    '0.5 × λ_{opt}', 
    'λ_{opt}', 
    '1.5 × λ_{opt}', 
    '0.9 × (2/B)',
    '1.1 × (2/B) [Divergent]',
    '1.5 × (2/B) [Divergent]'
};

max_iter = 1000;
errors_collection = cell(length(lambdas), 1);

fprintf('Testing %d different lambda values:\n', length(lambdas));
for i = 1:length(lambdas)
    lambda = lambdas(i);
    [errors, ~] = landweber_iteration(T_matrix, y, x_groundtruth, lambda, max_iter);
    errors_collection{i} = errors;
    
    if lambda < lambda_safe_boundary
        status = 'Convergent';
    else
        status = 'DIVERGENT';
    end
    
    fprintf('  λ_%d = %.6f [%s]: Final error = %.6e\n', ...
        i, lambda, status, errors(end));
end
fprintf('\n');

% Plot comparison
figure('Position', [100, 100, 1200, 800]);

% Convergent cases
subplot(2, 1, 1);
hold on;
colors = lines(4);
for i = 1:4
    semilogy(errors_collection{i}, 'LineWidth', 2, 'Color', colors(i,:), ...
        'DisplayName', lambda_labels{i});
end
grid on;
legend('Location', 'northeast', 'FontSize', 10);
title('Convergent Behavior (λ within valid interval)', 'FontSize', 14);
xlabel('Iteration Number', 'FontSize', 12);
ylabel('Error ||x_k - x_{ground truth}||', 'FontSize', 12);
set(gca, 'FontSize', 11);
hold off;

% Divergent cases
subplot(2, 1, 2);
hold on;
for i = 5:6
    semilogy(errors_collection{i}, 'LineWidth', 2, ...
        'DisplayName', lambda_labels{i});
end
grid on;
legend('Location', 'northwest', 'FontSize', 10);
title('Divergent Behavior (λ exceeds 2/B)', 'FontSize', 14);
xlabel('Iteration Number', 'FontSize', 12);
ylabel('Error ||x_k - x_{ground truth}||', 'FontSize', 12);
set(gca, 'FontSize', 11);
ylim([1e-2, 1e10]);
hold off;

%% ========================================================================
%% TASK 5: Algorithm Comparison (F.2 vs F.3) and Size Scaling
%% ========================================================================

fprintf('=== TASK 5: Algorithm Comparison and Scaling ===\n\n');

% Test different matrix sizes
sizes = [20, 50, 100, 200];
num_sizes = length(sizes);

figure('Position', [100, 100, 1400, 900]);

for s = 1:num_sizes
    N_test = sizes(s);
    M_test = round(0.6 * N_test);  % Maintain 60% measurement ratio
    
    % Generate test problem
    T_test = randn(M_test, N_test);
    x_test = randn(N_test, 1);
    y_test = T_test * x_test;
    x_gt = pinv(T_test) * y_test;
    
    % Compute bounds
    R_test = T_test' * T_test;
    eigs_test = eig(R_test);
    B_test = max(eigs_test);
    A_test = min(eigs_test(eigs_test > 1e-10));
    
    % Optimal lambdas for each algorithm
    lambda_F2 = 2 / (A_test + B_test);          % Theorem F.2
    lambda_F3 = sqrt(2 / (A_test^2 + B_test^2)); % Theorem F.3
    
    % Run both algorithms
    [errors_F2, ~] = landweber_iteration(T_test, y_test, x_gt, lambda_F2, 500);
    [errors_F3, ~] = landweber_squared_iteration(T_test, y_test, x_gt, lambda_F3, 500);
    
    % Plot comparison
    subplot(2, 2, s);
    semilogy(errors_F2, 'LineWidth', 2, 'DisplayName', 'Theorem F.2 (R)');
    hold on;
    semilogy(errors_F3, '--', 'LineWidth', 2, 'DisplayName', 'Theorem F.3 (R²)');
    grid on;
    legend('Location', 'northeast');
    title(sprintf('Size: %d×%d, κ≈%.2f', M_test, N_test, sqrt(B_test/A_test)));
    xlabel('Iteration');
    ylabel('Error');
    set(gca, 'FontSize', 10);
    hold off;
    
    fprintf('Size %dx%d: κ(T) ≈ %.4f\n', M_test, N_test, sqrt(B_test/A_test));
    fprintf('  F.2 final error: %.6e (λ=%.6f)\n', errors_F2(end), lambda_F2);
    fprintf('  F.3 final error: %.6e (λ=%.6f)\n', errors_F3(end), lambda_F3);
end

sgtitle('Algorithm Comparison: F.2 vs F.3 at Different Scales', 'FontSize', 16);

%% ========================================================================
%% FUNCTION DEFINITIONS (Task 4)
%% ========================================================================

function [errors, x_final] = landweber_iteration(T_matrix, y, x_groundtruth, lambda, max_iter)
    % LANDWEBER_ITERATION - Implements Theorem F.2 (Věta F.2)
    %
    % Inputs:
    %   T_matrix      - Measurement matrix (M x N)
    %   y             - Observation vector (M x 1)
    %   x_groundtruth - Ground truth solution for error tracking
    %   lambda        - Relaxation parameter (must be in (0, 2/||R||))
    %   max_iter      - Maximum number of iterations
    %
    % Outputs:
    %   errors        - Vector of errors ||x_k - x_groundtruth|| at each iteration
    %   x_final       - Final solution estimate
    
    [~, N] = size(T_matrix);
    
    % Define forward and adjoint operators
    T = @(x) T_matrix * x;
    T_adj = @(y) T_matrix' * y;
    
    % Initialize with zero (blank canvas)
    x_k = zeros(N, 1);
    errors = zeros(max_iter, 1);
    
    % Iterative reconstruction
    for k = 1:max_iter
        % Apply forward operator
        Tx = T(x_k);
        
        % Compute residual (error)
        residual = y - Tx;
        
        % Update step (Landweber iteration formula)
        correction = T_adj(residual);
        x_k = x_k + lambda * correction;
        
        % Track error
        errors(k) = norm(x_k - x_groundtruth);
    end
    
    x_final = x_k;
end

function [errors, x_final] = landweber_squared_iteration(T_matrix, y, x_groundtruth, lambda, max_iter)
    % LANDWEBER_SQUARED_ITERATION - Implements Theorem F.3 (Věta F.3)
    %
    % This uses R² instead of R for potentially more stable convergence
    % at the cost of slower convergence rate.
    %
    % Inputs: Same as landweber_iteration
    % Outputs: Same as landweber_iteration
    
    [~, N] = size(T_matrix);
    
    % Compute R = T'*T
    R = T_matrix' * T_matrix;
    
    % Initialize
    x_k = zeros(N, 1);
    errors = zeros(max_iter, 1);
    
    % Initial step
    x_1 = lambda^2 * R * (T_matrix' * y);
    x_k = x_1;
    
    % Iterative reconstruction
    for k = 1:max_iter
        % Update using R² formulation
        x_k = x_k + x_1 - lambda^2 * R^2 * x_k;
        
        % Track error
        errors(k) = norm(x_k - x_groundtruth);
    end
    
    x_final = x_k;
end

fprintf('\n=== Analysis Complete ===\n');
fprintf('All tasks completed successfully!\n');
