function [errors, x_final] = mod_f3(T_matrix, y, x_groundtruth, lambda, max_iter, tol)
    % Implementation of Theorem F.3
    %
    % Inputs:
    %   T_matrix      - The operator
    %   y             - Result
    %   x_groundtruth - The ground truth solution
    %   lambda        - Relaxation parameter
    %   max_iter      - number of iterations
    %   tol           - Tolerance for early stopping (e.g., 1e-9)
    %
    % Outputs:
    %   errors        - Vector of errors
    %   x_final       - Final solution estimate
    % This is more stable because of R^2 but at the cost of speed -- Theorem F3
    
    %[~, N] = size(T_matrix);
    T_adj = @(x) T_matrix' * y;
    
    % Compute R = T'*T
    R = T_matrix' * T_matrix;
    R_sq = R^2;
    
    % Initialize
    %x_k = zeros(N, 1);
    errors = zeros(max_iter, 1);
    
    % Initial step
    x_1 = lambda^2 * R * (T_adj(y));
    x_k = x_1;
    
    % Iterative reconstruction
    for k = 1:max_iter
        % Update using R_sq formulation
        x_k = x_k + x_1 - lambda^2 * R_sq * x_k;
        
        % Track error
        current_error = norm(x_k - x_groundtruth);
        errors(k) = current_error;

        % Checking the tolerance
        if current_error < tol
            errors = errors(1:k); % Trim the empty trailing zeros
            break; % Exit the loop immediately!
        end
    end
    
    x_final = x_k;
end
