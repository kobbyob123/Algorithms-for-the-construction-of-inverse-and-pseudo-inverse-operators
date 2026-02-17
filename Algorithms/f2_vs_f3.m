% Algorithm Comparison

fprintf('Comparing Algorithms\n\n');

% Test different matrix sizes
sizes = [20, 50, 100, 200];
num_sizes = length(sizes);

figure('Position', [100, 100, 1400, 900]);

for s = 1 : num_sizes
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
    lambda_F2 = 2 / (A_test + B_test);          % from F.2
    lambda_F3 = sqrt(2 / (A_test^2 + B_test^2)); % from F.3

    [errors_F2, ~] = f2(T_test, y_test, x_gt, lambda_F2, 500);
    [errors_F3, ~] = f3(T_test, y_test, x_gt, lambda_F3, 500);
    
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
    
    fprintf('Size %dx%d: K(T) ~ %.4f\n', M_test, N_test, sqrt(B_test/A_test));
    fprintf('  F.2 final error: %.6e (lam=%.6f)\n', errors_F2(end), lambda_F2);
    fprintf('  F.3 final error: %.6e (lam=%.6f)\n', errors_F3(end), lambda_F3);
end

sgtitle('Algorithm Comparison: F.2 vs F.3 at Different Scales', 'FontSize', 16);
