% Comparing LSMR with F2 and F3 -------------------- v2

rng(42);

% Setting up the parameters for LSMR --------------------
tol = 1e-5;
lambda = 0;         % no reg.
atol = tol;
btol = tol;
conlim = 1e12;      
itnlim = 2000;
localSize = 0;      
show = false;       

% Setting Parameters for Matrices --------------------
N_test = 100;
%M_test = round(0.6 * N_test);  % Maintain 60% measurement ratio (Tall Matrices)
M_test = 200;

% Generate test problem --------------------
T_test = randn(M_test, N_test);
x_test = randn(N_test, 1);
y_test = T_test * x_test;
x_gt = pinv(T_test) * y_test;
    
% Compute bounds for F2 and F3 --------------------
R_test = T_test' * T_test;
eigs_test = eig(R_test);
B_test = max(eigs_test);
A_test = min(eigs_test(eigs_test > 1e-10));
    
% Optimal lambdas for each algorithm --------------------
lambda_F2 = 2 / (A_test + B_test);          % from f2
lambda_F3 = sqrt(2 / (A_test^2 + B_test^2)); % from f3

% Running the Algorithms -----------------------
[errors_F2, ~] = mod_f2(T_test, y_test, x_gt, lambda_F2, itnlim, tol);
[errors_F3, ~] = mod_f3(T_test, y_test, x_gt, lambda_F3, itnlim, tol);
[x_final, errors_lsmr, istop, itn] = mod_lsmr(T_test, y_test, lambda, atol, btol, conlim, itnlim, localSize, show, x_gt);

% Displaying the Results -----------------------
fprintf('Iterations to reach %g tolerance:\n', tol);
fprintf('LSMR: %d iterations\n', length(errors_lsmr));
fprintf('F2:   %d iterations\n', length(errors_F2));
fprintf('F3:   %d iterations\n', length(errors_F3));

% Plotting convergence comparison -----------------------

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
