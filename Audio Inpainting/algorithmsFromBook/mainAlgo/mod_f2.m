% Second Version of the F2 Algorithm using Tolerances to reach a desired
% error level

function [errors, x_final] = mod_f2(T_matrix, y, x_groundtruth, lambda, max_iter, tol)
    % Implementation of Theorem F.2
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
    
    x_k = zeros(N, 1);
    errors = zeros(max_iter, 1); % prevent re-initialization cebause it wastes time
    
    for k = 1 : max_iter
        Tx = T(x_k); 
        residual = y - Tx;
        correction = T_adj(residual);
        x_k = x_k + lambda * correction;
        
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
