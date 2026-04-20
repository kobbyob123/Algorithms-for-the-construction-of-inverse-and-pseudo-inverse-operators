% Objective function coefficients (MATLAB's linprog minimizes, so negate)
f = [-5; -4];

% Inequality constraints A*x <= b
A = [1, 2;    % x1 + 2x2 <= 6
     2, -1;   % 2x1 - x2 <= 4
     5, 3];   % 5x1 + 3x2 <= 15

b = [6; 4; 15];

% Lower bounds (x1, x2 >= 0)
lb = [0; 0];

% Upper bounds (no upper bounds)
ub = [];

% Solve using linprog
[x, fval, exitflag, output] = linprog(f, A, b, [], [], lb, ub);

% Display results
if exitflag == 1
    fprintf('Optimization successful!\n');
    fprintf('Optimal x1: %.4f\n', x(1));
    fprintf('Optimal x2: %.4f\n', x(2));
    fprintf('Maximized Z: %.4f\n', -fval);
else
    fprintf('Optimization failed.\n');
end