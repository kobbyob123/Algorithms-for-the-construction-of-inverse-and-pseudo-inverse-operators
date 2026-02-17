%% Comparison of F.2 and F.3 with Theoretical Bounds

clear; clc; close all;
N = 100; M = 0.6 * N;
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
    
    % increment Size
    N = N * 2;
    M = M * 2;

