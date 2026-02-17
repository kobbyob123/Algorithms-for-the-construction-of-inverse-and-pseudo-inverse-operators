%% Define problem dimensions
N = 50;  % Signal size
M = 30;  % Observation size (M < N means underdetermined)

% Define a random operator to start 
T_matrix = randn(M, N);

% Define true signal
x_true = randn(N, 1);

% Define observed signal
y = T_matrix * x_true;

% Define ground truth using pseudoinverse
x_groundtruth = pinv(T_matrix) * y;

% Compute spectral bounds for lambda selection
R_matrix = T_matrix' * T_matrix;
eigenvalues = eig(R_matrix);
B = max(eigenvalues);  % Upper bound
A = min(eigenvalues(eigenvalues > 1e-10));  % Lower bound (exclude near-zero)

%% Testing different lambdas

fprintf('Test for different values of lambda\n\n');

% Define lambda values to test
lambda_optimal = 2/(A + B);
lambda_safe_boundary = 2/B;
new_lmda = 0.5 * lambda_optimal + 0.5*lambda_safe_boundary;
lambdas = [0.5 * lambda_optimal, lambda_optimal, 1.5 * lambda_optimal, new_lmda , 0.9 * lambda_safe_boundary, 1.1 * lambda_safe_boundary, 1.5 * lambda_safe_boundary];

lambda_labels = {'0.5 × lmda_{opt}', 'lmda_{opt}', '1.5 × lmda_{opt}', 'new_lambda','0.9 × (2/B)', '1.1 × (2/B)', '1.5 × (2/B)'};

max_iter = 1000;
errors_collection = cell(length(lambdas), 1);

fprintf('Testing %d different lambda values:\n', length(lambdas));
for i = 1:length(lambdas)
    lambda = lambdas(i);
    [errors, ~] = f2(T_matrix, y, x_groundtruth, lambda, max_iter); % here
    errors_collection{i} = errors;
    
    if lambda < lambda_safe_boundary
        status = 'Convergent';
    else
        status = 'DIVERGENT';
    end
    
    fprintf('lmda_%d = %.6f [%s]: Final error = %.6e\n', ...
        i, lambda, status, errors(end));
end
fprintf('\n');

% Plot comparison
figure('Position', [100, 100, 1200, 800]);

% Convergent cases
subplot(2, 1, 1);
colors = lines(5);
for i = 1:5
    semilogy(errors_collection{i}, 'LineWidth', 2, 'Color', colors(i,:), ...
        'DisplayName', lambda_labels{i});
hold on;
end

grid on;
legend('Location', 'northeast', 'FontSize', 10);
title('Convergent Behavior (lmda within valid interval)', 'FontSize', 14);
xlabel('Iteration Number', 'FontSize', 12);
ylabel('Error ||x_k - x_{ground truth}||', 'FontSize', 12);
set(gca, 'FontSize', 11);
hold off;

% Divergent cases
subplot(2, 1, 2);
hold on;
for i = 5:6
    semilogy(errors_collection{i}, 'LineWidth', 2, ...
        'DisplayName', lambda_labels{i});
end
grid on;
legend('Location', 'northwest', 'FontSize', 10);
title('Divergent Behavior (lmda exceeds 2/B)', 'FontSize', 14);
xlabel('Iteration Number', 'FontSize', 12);
ylabel('Error ||x_k - x_{ground truth}||', 'FontSize', 12);
set(gca, 'FontSize', 11);
ylim([1e-2, 1e10]);
hold off;
