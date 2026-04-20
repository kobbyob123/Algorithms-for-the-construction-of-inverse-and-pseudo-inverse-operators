% Test Script for Modified LSMR (error checking included)
clc; clear; close all;

rng(42);

m = 30;
n = 10;
A = randn(m, n);         
x_true = randn(n, 1);    
b = A * x_true;

% Setting up the parameters for LSMR
lambda = 0;              % no regularization
atol = 0;            % Set very tight tolerances to prevent early stopping
btol = 0;            % so we can see the full error curve
conlim = 1e12;           % Condition number limit
itnlim = 500;            % Maximum number of iterations to plot
localSize = 0;           % no reorthogonalization
show = false;            % Verbose

% Call the modified LSMR function
[x_final, errors_lsmr, istop, itn] = mod_lsmr(A, b, lambda, atol, btol, conlim, itnlim, localSize, show, x_true);

% Plotting "convergence profile" using semilogy
figure;
semilogy(1:itn, errors_lsmr, 'LineWidth', 1.5, 'DisplayName', 'LSMR');
grid on;
title('Convergence Profile of LSMR');
xlabel('Iteration count (k)');
ylabel('Distance to Ground Truth: ||x_k - x_{true}||');
legend('show');
