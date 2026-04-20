clear;
clc;
close all;

% --- Setup ---
% Define the matrix sizes to test
sizes = [100, 500, 1000, 2000, 4000, 6000];
% num_trials = 3; 
times_pinv = zeros(1, length(sizes));
times_mldivide = zeros(1, length(sizes));
times_cgls = zeros(1, length(sizes));

fprintf('Starting CGLS benchmark...\n');

% --- Main Loop ---
for i = 1:length(sizes)
    n = sizes(i);
    fprintf('Testing matrix size: %d x %d\n', n, n);
    
    % Create random matrix and vector
    A = rand(n, n);
    b = rand(n, 1);
    
    % --- Time pinv ---
    f_pinv = @() pinv(A) * b; 
    times_pinv(i) = timeit(f_pinv, 1);
    
    % --- Time mldivide (\) ---
    f_mldivide = @() A \ b;
    times_mldivide(i) = timeit(f_mldivide, 1);
    
    % --- Time cgls ---
    % Parameters: A, b, shift=0, tol=1e-6, maxit=100, prnt=0
    f_cgls = @() cgls(A, b, 0, 1e-6, 100, 0); 
    times_cgls(i) = timeit(f_cgls, 1);
end

fprintf('Benchmark complete.\n');

% --- Plotting ---
figure;
% semi-log for logarithm
plot(sizes, times_pinv, 'r-o', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot(sizes, times_mldivide, 'b-s', 'LineWidth', 2, 'MarkerSize', 8);
plot(sizes, times_cgls, 'g-^', 'LineWidth', 2, 'MarkerSize', 8);
hold off;

grid on;
legend('pinv(A) * b', 'A \ b (mldivide)', 'cgls(A, b)', 'Location', 'northwest');
xlabel('Matrix Size (n x n)');
ylabel('Computation Time (seconds)');
title('Benchmark: CGLS vs Direct Solvers');
set(gca, 'FontSize', 12);