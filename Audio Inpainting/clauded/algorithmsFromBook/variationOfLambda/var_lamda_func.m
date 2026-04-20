%% Lamda Variation

clear; clc; close all;

%% Define problem dimensions
N = 50;  % Signal size
M = 30;  % Observation size 

% Define a random operator to start 
rng(42); % For reproducibility
T_matrix = randn(M, N);

% Define true signal
x_true = randn(N, 1);

% Define observed signal
y = T_matrix * x_true;

% Define ground truth using pseudoinverse - pinv
x_groundtruth = pinv(T_matrix) * y;

% Compute spectral bounds for lambda selection
R_matrix = T_matrix' * T_matrix;
eigenvalues = eig(R_matrix);
B = max(eigenvalues);  % Upper bound
A = min(eigenvalues(eigenvalues > 1e-10));  % Lower bound (exclude near-zero)

fprintf('Matrix Properties:\n');
fprintf('  Spectral bounds: A = %.6f, B = %.6f\n', A, B);
fprintf('  Condition number: κ ≈ %.4f\n\n', sqrt(B/A));

%% Testing different lambdas

fprintf('=== Lambda Parameter Study ===\n\n');

% Define lambda values to test
lambda_optimal = 2/(A + B);
lambda_safe_boundary = 2/B;
new_lmda = 0.5 * lambda_optimal + 0.5 * lambda_safe_boundary;

lambdas = [
    % add little offsets to analyse sensitivity
    0.5 * lambda_optimal,
    lambda_optimal,
    1.5 * lambda_optimal,
    new_lmda,
    0.9 * lambda_safe_boundary,
    1.1 * lambda_safe_boundary,
    1.5 * lambda_safe_boundary
];

lambda_labels = {
    '0.5 × lam_{opt}',
    'lam_{opt}',
    '1.5 × lam_{opt}',
    'new lam',
    '0.9 × (2/B)',
    '1.1 × (2/B)',
    '1.5 × (2/B)'
};

max_iter = 1000;  % is it enough?

%% Prepare figure with axis handles befre the loop
figure('Position', [100, 100, 1200, 800]);

% Create axis handle for CONVERGENT cases
ax_convergent = subplot(2, 1, 1);
hold(ax_convergent, 'on');
grid(ax_convergent, 'on');
title(ax_convergent, 'Convergent Behavior (lam < 2/B)', 'FontSize', 14);
xlabel(ax_convergent, 'Iteration Number', 'FontSize', 12);
ylabel(ax_convergent, 'Error ||x_k - x_{groundtruth}||', 'FontSize', 12);
set(ax_convergent, 'FontSize', 11, 'YScale', 'log');

% Create axis handle for DIVERGENT cases
ax_divergent = subplot(2, 1, 2);
hold(ax_divergent, 'on');
grid(ax_divergent, 'on');
title(ax_divergent, 'Divergent Behavior (λ ≥ 2/B)', 'FontSize', 14);
xlabel(ax_divergent, 'Iteration Number', 'FontSize', 12);
ylabel(ax_divergent, 'Error ||x_k - x_{groundtruth}||', 'FontSize', 12);
set(ax_divergent, 'FontSize', 11, 'YScale', 'log');
ylim(ax_divergent, [1e-2, 1e10]);

% Prepare colors for each category
num_convergent = sum(lambdas < lambda_safe_boundary);
num_divergent = sum(lambdas >= lambda_safe_boundary);
colors_conv = lines(num_convergent);
colors_div = lines(num_divergent);

conv_counter = 1;
div_counter = 1;

%% Loop through lambdas and assign to correct subplot

fprintf('Testing %d different lambda values:\n', length(lambdas));
fprintf('Stability boundary: lam_max = 2/B = %.6f\n', lambda_safe_boundary);
fprintf('Optimal lambda: lam_opt = 2/(A+B) = %.6f\n\n', lambda_optimal);

for i = 1:length(lambdas)
    lambda = lambdas(i);
    
    % Runng the iteration algorithm
    [errors, ~] = f2(T_matrix, y, x_groundtruth, lambda, max_iter);
    
    % AUTOMATIC DECISION: Check if convergent or divergent
    if lambda < lambda_safe_boundary
        % CONVERGENT case
        target_axis = ax_convergent;
        status = 'Convergent';
        line_color = colors_conv(conv_counter, :);
        conv_counter = conv_counter + 1;
    else
        % DIVERGENT case
        target_axis = ax_divergent;
        status = 'DIVERGENT';
        line_color = colors_div(div_counter, :);
        div_counter = div_counter + 1;
    end
    
    % Plot to the SELECTED axis
    semilogy(target_axis, 1:max_iter, errors, 'LineWidth', 2, ...
        'Color', line_color, ...
        'DisplayName', sprintf('%s (lam=%.4f)', lambda_labels{i}, lambda));
    
    % Print status
    fprintf('  [%d] %s: lam = %.6f [%s] → Final error = %.6e\n', ...
        i, lambda_labels{i}, lambda, status, errors(end));
end

% Finalize legends on both subplots
legend(ax_convergent, 'Location', 'northeast', 'FontSize', 10);
legend(ax_divergent, 'Location', 'northwest', 'FontSize', 10);

fprintf('\n=== Analysis Complete ===\n');
