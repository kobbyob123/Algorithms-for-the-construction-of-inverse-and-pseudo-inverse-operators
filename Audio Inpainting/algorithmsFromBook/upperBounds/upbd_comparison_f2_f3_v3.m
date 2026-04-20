%% Comparison of F.2 and F.3 with Theoretical Bounds - Multiple Runs

clear; clc; close all;
N = 100; M = ceil(0.6 * N);
rng(42);

% Problem Setup
for i = 1 : 10
    T = randn(M, N);
    x_gt = randn(N, 1);
    y = T * x_gt;
    x_sol = pinv(T) * y;
    
    % Setting Spectral Properties
    R = T' * T;
    eigs = eig(R);
    B = max(eigs);
    val = 1e-10;
    A = min(eigs(eigs > val));
    
    % Optimal Parameters
    lam_f2 = 2 / (A + B);
    lam_f3 = sqrt(2 / (A^2 + B^2));
    
    r_f2 = (B - A) / (B + A);
    r_f3 = (B^2 - A^2) / (B^2 + A^2);
    
    % Run Iterations
    max_iter = 500;
    [err_f2, ~] = f2(T, y, x_sol, lam_f2, max_iter);
    [err_f3, ~] = f3(T, y, x_sol, lam_f3, max_iter);

    % Compute Theoretical Bounds
    bound_f2 = (r_f2 .^ (0:max_iter-1)) * norm(x_sol);
    bound_f3 = (r_f3 .^ (0:max_iter-1)) * norm(x_sol);
    
    % Visualization
    figure('Position', [100, 100, 1300, 500]);
    
    % Subplot F.2 Analysis
    subplot(1, 2, 1);
    semilogy(err_f2, 'b-', 'LineWidth', 2, 'DisplayName', 'Observed Error (F.2)');
    hold on;
    semilogy(bound_f2, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Theoretical Bound');
    grid on;
    title('Theorem F.2: Accuracy vs Bound');
    xlabel('Iteration'); ylabel('Error');
    legend('Location', 'northeast');
    
    % Subplot F.3 Analysis
    subplot(1, 2, 2);
    semilogy(err_f3, 'g-', 'LineWidth', 2, 'DisplayName', 'Observed Error (F.3)');
    hold on;
    semilogy(bound_f3, 'm--', 'LineWidth', 1.5, 'DisplayName', 'Theoretical Bound');
    grid on;
    title('Theorem F.3: Accuracy vs Bound');
    xlabel('Iteration'); ylabel('Error');
    legend('Location', 'northeast');
    
    sgtitle(sprintf('Run %d: N=%d, M=%d (k ~ %.2f)', i, N, M, sqrt(B/A)), 'FontSize', 16);
    
    % Save the figure as PNG
    %filename = sprintf('comparison_run_%02d_N%d.png', i, N);
    %saveas(gcf, filename);
    %fprintf('Saved: %s\n', filename);
    
    % Close the figure to free memory
    %close(gcf);
    
    % Increment Size
    N = ceil(N * 1.2);
    M = ceil(M * 1.2);
end

fprintf('All runs completed!\n');
