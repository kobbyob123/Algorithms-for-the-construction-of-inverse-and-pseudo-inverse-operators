% --- STEP 1: SETUP THE PROBLEM ---
clear; clc;

% Define dimensions
N = 50;  
M = 30;

% Create a random "Masking Operator" (Matrix T)
T_matrix = randn(M, N); 

% Define the "True Signal"
x_true = randn(N, 1);

% Create the "Observed Signal" y (The corrupted version)
y = T_matrix * x_true;

% Define T (Forward) and T* (Adjoint)
T     = @(x) T_matrix * x;
T_adj = @(y) T_matrix' * y; % Adjoint

%% Calculate R = T*T to find the speed limit (Lambda)
R_matrix = T_matrix' * T_matrix;
max_eigenvalue = max(eig(R_matrix)); % This is ||R||
B = max_eigenvalue;
disp(B);

% Set Lambda (Relaxation Parameter)
opti_lambda = 1.9 / B;
lambdas = [opti_lambda, 1.1 * opti_lambda, ];

%% Initialization: Start with a blank canvas (vectors of zeros)
% The formula implicitly starts the sum at n = 0.
x_k = zeros(N, 1); 

% Number of iterations (The "Sum" from n=0 to infinity)
max_iter = 1000;
errors = zeros(max_iter, 1);

fprintf('Starting Iteration (Věta F.2)...\n');

for k = 1 : max_iter
    
    % 1. Apply the Painter (Forward Operator)
    Tx = T(x_k);
    
    % 2. Find the Residual (The Error)
    residual = y - Tx;
    
    % 3. Update Step
    % x_new = x_old + lambda * T*(error)
    correction = T_adj(residual);
    x_k = x_k + lambda * correction;
    
    % Track error
    errors(k) = norm(residual);
end

%% CHECK RESULTS
fprintf('Final Error: %.6f\n', errors(end));

% Comparing with pinv
x_pinv = pinv(T_matrix) * y;
difference = norm(x_k - x_pinv);

fprintf('Difference between Věta F.2 and MATLAB pinv: %.6f\n', difference);

%% Plot the convergence
figure;
semilogy(errors, 'LineWidth', 2);
title('Convergence of Věta F.2 (Landweber)');
xlabel('Iteration (n)');
ylabel('Error ||y - Tx||');
grid on;
