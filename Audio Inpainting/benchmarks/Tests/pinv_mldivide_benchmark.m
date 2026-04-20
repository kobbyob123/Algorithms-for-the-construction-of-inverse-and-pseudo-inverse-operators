clear;
clc;
close all;

% --- Setup ---
% Define the matrix sizes to test
sizes = [100, 500, 1000, 2000, 4000, 6000]; 
num_trials = 3; % Run a few times to get an average
times_pinv = zeros(1, length(sizes));
times_mldivide = zeros(1, length(sizes));

fprintf('Starting benchmark...\n');

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
end

fprintf('Benchmark complete.\n');

% --- Plotting ---
figure;
plot(sizes, times_pinv, 'r-o', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot(sizes, times_mldivide, 'b-s', 'LineWidth', 2, 'MarkerSize', 8);
hold off;

grid on;
legend('pinv(A) * b', 'A \ b (mldivide)', 'Location', 'northwest');
xlabel('Matrix Size (n x n)');
ylabel('Computation Time (seconds)');
title('Benchmark of (Pseudo)Inverse Solvers');
set(gca, 'FontSize', 12);