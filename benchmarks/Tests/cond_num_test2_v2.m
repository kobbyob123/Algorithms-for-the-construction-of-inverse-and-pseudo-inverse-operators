% clc; clear; close all;

rng(42);
sizes = [900 950 1000 1100 1200];
tol = 1e-9;
itnlim = 5000;

% Array of condition numbers to test
condition_numbers = 1:5:22;

num_sizes = length(sizes);
num_kappas = length(condition_numbers);

% Pre-allocate 2D arrays (sizes x condition_numbers)
iters_LSMR = zeros(num_sizes, num_kappas);
iters_F2   = zeros(num_sizes, num_kappas);
iters_F3   = zeros(num_sizes, num_kappas);

for i = 1:num_sizes
    M = sizes(i) * 1.5;
    N = sizes(i);
    for j = 1:num_kappas
        kappa = condition_numbers(j);
        fprintf('Testing matrix of size: %d x %d\n', M, N);
        fprintf('Testing Condition Number: %d\n', kappa);

        % 1. Generate matrix with specific condition number
        T_test = generate_matrix(M, N, kappa);
        x_test = randn(N, 1);
        y_test = T_test * x_test;
        x_gt = pinv(T_test) * y_test;

        % 2. Calculate optimal lambdas
        R_test = T_test' * T_test;
        eigs_test = eig(R_test);
        % B_test = max(eigs_test);
        % A_test = min(eigs_test(eigs_test > 1e-10));

         % from example 5.13
        B_test = svds(T_test, 1, 'largest')^2;
        A_test = svds(T_test, 1, 'smallestnz')^2;
        lambda_F2 = 2 / (A_test + B_test);
        lambda_F3 = sqrt(2 / (A_test^2 + B_test^2));

        % Run algorithms and record iterations
        [~, errors_lsmr, ~, ~] = mod_lsmr(T_test, y_test, 0, tol, tol, 1e7, itnlim, 0, false, x_gt);
        iters_LSMR(i, j) = length(errors_lsmr);

        [errors_F2, ~] = mod_f2(T_test, y_test, x_gt, lambda_F2, itnlim, tol);
        iters_F2(i, j) = length(errors_F2);

        [errors_F3, ~] = mod_f3(T_test, y_test, x_gt, lambda_F3, itnlim, tol);
        iters_F3(i, j) = length(errors_F3);
    end
end

% Build a flat table with one row per (size, kappa) pair
n_rows = num_sizes * num_kappas;
col_size   = zeros(n_rows, 1);
col_kappa  = zeros(n_rows, 1);
col_lsmr   = zeros(n_rows, 1);
col_f2     = zeros(n_rows, 1);
col_f3     = zeros(n_rows, 1);

row = 1;
for i = 1:num_sizes
    for j = 1:num_kappas
        col_size(row)  = sizes(i);
        col_kappa(row) = condition_numbers(j);
        col_lsmr(row)  = iters_LSMR(i, j);
        col_f2(row)    = iters_F2(i, j);
        col_f3(row)    = iters_F3(i, j);
        row = row + 1;
    end
end

ResultsTable = table(col_size, col_kappa, col_lsmr, col_f2, col_f3, ...
    'VariableNames', {'Matrix_Size', 'Condition_Number', 'LSMR_Iters', 'F2_Iters', 'F3_Iters'});
disp(ResultsTable);

% Plot one subplot per matrix size, with all 3 algorithms per subplot
figure('Position', [100, 100, 1400, 800]);
colors = lines(3);

for i = 1:num_sizes
    subplot(2, 3, i);
    plot(condition_numbers, iters_F2(i, :),   '-o', 'LineWidth', 1.5, 'Color', colors(1,:), 'DisplayName', 'F2');
    hold on;
    plot(condition_numbers, iters_F3(i, :),   '-^', 'LineWidth', 1.5, 'Color', colors(2,:), 'DisplayName', 'F3');
    plot(condition_numbers, iters_LSMR(i, :), '-s', 'LineWidth', 1.5, 'Color', colors(3,:), 'DisplayName', 'LSMR');
    grid on;
    xlabel('Condition Number (\kappa)');
    ylabel(sprintf('Iterations to reach %g', tol));
    title(sprintf('Matrix size: %d \\times %d', sizes(i), round(sizes(i) * 1.5)));
    legend('show', 'Location', 'northwest');
    hold off;
end

sgtitle('Effect of Condition Number on Convergence Speed', 'FontSize', 14);
