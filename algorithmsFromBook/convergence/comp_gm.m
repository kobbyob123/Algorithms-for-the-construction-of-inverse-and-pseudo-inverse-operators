% Clear workspace
clear; clc; close all;

% --- 1. Setup the Problem ---
% Define problem size
m = 1000;  % Rows
n = 500;   % Cols (Overdetermined system for this test)

% Create a random matrix A
A_matrix = randn(m, n);

% Create a "Ground Truth" solution (x_true)
% This is what we want the algorithms to find.
x_true = randn(n, 1);

% Create the Right-Hand Side (b)
% b = A * x_true + some noise
noise_level = 1e-4;
b = A_matrix * x_true + noise_level * randn(m, 1);

% Parameters for solvers
max_iter = 1000;
lambda   = 0;      % Regularization (0 for standard LS)
atol     = 1e-9;   % Tolerance
btol     = 1e-9;

% --- 2. Define the Operator (AFUN) ---
% LSMR prefers a function handle instead of a matrix for large problems.
% This mimics how you will use it for Audio Inpainting later.
A_fun = @(x, mode) mode_handler(A_matrix, x, mode);

% --- 3. Run Solvers ---

fprintf('Running f2 (Landweber)...\n');
% Note: f2/f3 usually require a small lambda/step-size for stability
step_size = 1 / (norm(A_matrix)^2); % Optimal step for Landweber
[err_f2, x_f2] = f2(A_matrix, b, x_true, step_size, max_iter);

fprintf('Running f3 (Landweber with R^2)...\n');
[err_f3, x_f3] = f3(A_matrix, b, x_true, step_size, max_iter);

fprintf('Running LSMR (Tracked)...\n');
% Call the modified LSMR with x_true at the end
[x_lsmr, istop, itn, normr, normAr, normA, condA, normx, err_lsmr] ...
    = lsmr_2(A_fun, b, lambda, atol, btol, [], max_iter, [], false, x_true);

% --- 4. Plot Comparison ---
figure('Name', 'Convergence Comparison');
semilogy(err_f2, 'LineWidth', 2, 'DisplayName', 'f2 (Landweber)');
hold on;
semilogy(err_f3, 'LineWidth', 2, 'DisplayName', 'f3 (Landweber R^2)');
semilogy(err_lsmr, 'LineWidth', 2, 'DisplayName', 'LSMR');
grid on;
xlabel('Iteration');
ylabel('Error ||x_k - x_{true}||');
title('Convergence Speed: LSMR vs Landweber');
legend;

% --- Helper Function for LSMR Operator ---
function y = mode_handler(A, x, mode)
    if mode == 1
        y = A * x;  % Ax
    else
        y = A' * x; % A'x
    end
end
