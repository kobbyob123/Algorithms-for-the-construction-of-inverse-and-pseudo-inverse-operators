function A = generate_matrix(m, n, cond_num)
    % Generate random orthogonal matrices U and V
    [U, ~, ~] = svd(randn(m, min(m,n)), 'econ');
    [V, ~, ~] = svd(randn(n, min(m,n)), 'econ');
    
    % Create singular values linearly spaced from 1 down to 1/cond_num
    s_values = linspace(1, 1/cond_num, min(m,n));
    S = diag(s_values);
    
    % Construct the matrix
    A = U * S * V';
end
