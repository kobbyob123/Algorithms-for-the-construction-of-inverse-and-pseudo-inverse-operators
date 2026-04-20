% Comparing LSMR with F2 and F3 ------ v1

% ------------ Setting up the parameters for LSMR --------------------
lambda = 0;              % no regularization
atol = 1e-12;            % Set very tight tolerances to prevent early stopping
btol = 1e-12;            % so we can see the full error curve
conlim = 1e12;           % Condition number limit
itnlim = 500;            % Maximum number of iterations to plot
localSize = 0;           % no reorthogonalization
show = false;            % Set to true if you want to see the terminal prints

% ------------- Setting Parameters for Matrices --------------------
N_test = 2000;
M_test = round(0.6 * N_test);  % Maintain 60% measurement ratio
    
% Generate test problem
T_test = randn(M_test, N_test);
x_test = randn(N_test, 1);
y_test = T_test * x_test;
x_gt = pinv(T_test) * y_test; % actual true solution
    
% Compute bounds for F2 and F3
R_test = T_test' * T_test;
eigs_test = eig(R_test);
B_test = max(eigs_test);
A_test = min(eigs_test(eigs_test > 1e-10));
    
% Optimal lambdas for each algorithm
lambda_F2 = 2 / (A_test + B_test);          % from F.2
lambda_F3 = sqrt(2 / (A_test^2 + B_test^2)); % from F.3

% -------------- Running The Algorithm -------------------------------
[errors_F2, ~] = f2(T_test, y_test, x_gt, lambda_F2, itnlim);
[errors_F3, ~] = f3(T_test, y_test, x_gt, lambda_F3, itnlim);
[x_final, errors_lsmr, istop, itn] = mod_lsmr(T_test, y_test, lambda, atol, btol, conlim, itnlim, localSize, show, x_gt);

% -------------- Plotting convergence comparison -----------------------

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
