function [errors, x_final] = f2(T_matrix, y, x_groundtruth, lambda, max_iter)
    % Trial Implementation of Theorem F.2
    %
    % Inputs:
    %   T_matrix      - The operator
    %   y             - Result
    %   x_groundtruth - The ground truth solution
    %   lambda        - Relaxation parameter
    %   max_iter      - number of iterations
    %
    % Outputs:
    %   errors        - Vector of errors
    %   x_final       - Final solution estimate
    
    [~, N] = size(T_matrix);
    
    % Define forward and adjoint operators
    T = @(x) T_matrix * x;
    T_adj = @(y) T_matrix' * y;
    
    % Initialize with zero
    x_k = zeros(N, 1);
    errors = zeros(max_iter, 1);
    
    % Iterative reconstruction
    
    for k = 1 : max_iter
        Tx = T(x_k);
        
        % Computing residual (error)
        residual = y - Tx;
        
        % Update step
        correction = T_adj(residual);
        x_k = x_k + lambda * correction;
        
        % Track error
        errors(k) = norm(x_k - x_groundtruth);
    end
    
    % check on stopping criterion, tolerance, conditionals
    
    x_final = x_k;
end
