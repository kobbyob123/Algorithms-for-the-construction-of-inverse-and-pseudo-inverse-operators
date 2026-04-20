%clc; clear; close all;

% Set constants
N = 1000;
M = 1500; % Tall matrix
tol = 1e-9;
itnlim = 2000; % High limit to let them finish

% Array of condition numbers to test
condition_numbers = 1:8;

% Pre-allocate arrays to store the results
iters_LSMR = zeros(length(condition_numbers), 1);
iters_F2 = zeros(length(condition_numbers), 1);
iters_F3 = zeros(length(condition_numbers), 1);

for i = 1:length(condition_numbers)
    kappa = condition_numbers(i);
    fprintf('Testing Condition Number: %d\n', kappa);
    
    % 1. Generate matrix with specific condition number
    T_test = generate_matrix(M, N, kappa);
    x_test = randn(N, 1);
    y_test = T_test * x_test;
    x_gt = pinv(T_test) * y_test;
    
    % 2. Calculate optimal lambdas
    R_test = T_test' * T_test;
    eigs_test = eig(R_test);
    %B_test = max(eigs_test);
    %A_test = min(eigs_test(eigs_test > 1e-10));

    % from example 5.13
    B_test = svds(T_test, 1, 'largest')^2;
    A_test = svds(T_test, 1, 'smallestnz')^2;

    lambda_F2 = 2 / (A_test + B_test);
    lambda_F3 = sqrt(2 / (A_test^2 + B_test^2));
    
    % 3. Run algorithms and record the length of the error array (iterations)
    [~, errors_lsmr, ~, itn] = mod_lsmr(T_test, y_test, 0, tol, tol, 1e12, itnlim, 0, false, x_gt);
    iters_LSMR(i) = length(errors_lsmr);
    
    [errors_F2, ~] = mod_f2(T_test, y_test, x_gt, lambda_F2, itnlim, tol);
    iters_F2(i) = length(errors_F2);
    
    [errors_F3, ~] = mod_f3(T_test, y_test, x_gt, lambda_F3, itnlim, tol);
    iters_F3(i) = length(errors_F3);
end

%  Create and display a table
ResultsTable = table(condition_numbers', iters_LSMR, iters_F2, iters_F3, 'VariableNames', {'Condition_Number', 'LSMR_Iters', 'F2_Iters', 'F3_Iters'});
disp(ResultsTable);

figure;
plot(condition_numbers, iters_F2, 'b-o', 'LineWidth', 1.5, 'DisplayName', 'F2');
hold on;
plot(condition_numbers, iters_F3, 'g-^', 'LineWidth', 1.5, 'DisplayName', 'F3');
plot(condition_numbers, iters_LSMR, 'r-s', 'LineWidth', 1.5, 'DisplayName', 'LSMR');
grid on;
xlabel('Condition Number (\kappa)');
ylabel(sprintf('Iterations to reach %g', tol));
title('Effect of Condition Number on Convergence Speed');
legend('show', 'Location', 'northwest')
