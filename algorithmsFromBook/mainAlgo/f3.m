function [errors, x_final] = f3(T_matrix, y, x_groundtruth, lambda, max_iter)
    % Implements Theorem F.3
    %
    % This is more stable because of R^2 but at the cost of speed -- Theorem F3
    
    [~, N] = size(T_matrix);
    
    % Compute R = T'*T
    R = T_matrix' * T_matrix;
    
    % Initialize
    x_k = zeros(N, 1);
    errors = zeros(max_iter, 1);
    
    % Initial step
    x_1 = lambda^2 * R * (T_matrix' * y);
    x_k = x_1;
    
    % Iterative reconstruction
    for k = 1:max_iter
        % Update using R² formulation
        x_k = x_k + x_1 - lambda^2 * R^2 * x_k;
        
        % Track error
        errors(k) = norm(x_k - x_groundtruth);
    end
    
    x_final = x_k;
end
