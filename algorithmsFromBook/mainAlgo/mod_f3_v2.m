% Second Version of the F3 Algorithm using Tolerances to reach a desired
% error level

function [errors, x_final] = mod_f3(T_matrix, y, x_groundtruth, lambda, max_iter, tol)
    % Implementation of Theorem F.3 / Corollary F.5
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
    
    [~, N] = size(T_matrix);
    T = @(x) T_matrix * x;
    T_adj = @(y) T_matrix' * y;
    
    % --- 1. Mathematical Initialization (Corollary F.5) ---
    % The algorithm requires starting vector x_1 = \lambda^2 * R * T^* * y
    % Since R = T^* * T, this expands to x_1 = \lambda^2 * T^* * T * T^* * y
    T_y = T_adj(y);
    T_T_y = T(T_y);
    x_1 = (lambda^2) * T_adj(T_T_y);
    
    x_k = x_1; % The iterations natively start at x_1
    errors = zeros(max_iter, 1); % prevent re-initialization because it wastes time
    
    % --- 2. Iterative Reconstruction Loop ---
    for k = 1 : max_iter
        
        % Applying the squared operator R^2 x_k = T^* * T * T^* * T * x_k
        Tx = T(x_k);
        R_x = T_adj(Tx);
        T_R_x = T(R_x);
        R2_x = T_adj(T_R_x);
        
        % Update step (Corollary F.5): x_{k+1} = x_k + x_1 - \lambda^2 R^2 x_k
        x_k = x_k + x_1 - (lambda^2) * R2_x;
        
        % Calculate current error
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