%% Creating a "Fake" Audio Reconstruction Problem
clear; clc;

% m rows (equations), n cols (unknowns)
m = 1000; 
n = 1000; 
density = 0.05; % 5% non-zero elements (Sparse like audio matrices)
A = sprand(m, n, density);

function y = AFUN(x, v, A)
 if v == 1
     y = A*x;
 else
     y = A'*x;
 end
end

% Anonymous function handle for A
Afun = @(x, v) AFUN(x, v, A);

% 2. Create a "True" signal and a corrupted measurement 'b'
x_true = randn(n, 1);
b = A * x_true;

% Add some noise (simulating real-world audio corruption)
b = b + 0.01 * randn(m, 1);

%% Run LSMR
fprintf('Running LSMR...\n');
lambda = 0.1;
itnlim = 200;
show   = true;

tic;
[x_est, istop, itn, normr] = lsmr(Afun, b, lambda, [], [], [], itnlim, [], show);
time_lsmr = toc;

%% Analyze Results
fprintf('\n--- Results ---\n');
fprintf('Time taken:       %.4f seconds\n', time_lsmr);
fprintf('Iterations:       %d\n', itn);
fprintf('Final Residual:   %.4e\n', normr);
fprintf('Reconstruction Error: %.4e\n', norm(x_est - x_true)/norm(x_true));

if istop == 7
    fprintf('Warning: Reached maximum iterations before converging.\n');
else
    fprintf('Success: Converged successfully.\n');
end

