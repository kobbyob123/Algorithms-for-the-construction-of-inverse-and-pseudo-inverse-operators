% Create time axis
t = linspace(0, 10, 100);

% Experimental trend: decreasing effective porosity / transport
experimental = exp(-0.2 * t);

% Model assumption: constant porosity
model = ones(size(t));

figure('Position', [100, 100, 700, 400]);
plot(t, experimental, 'LineWidth', 3);
hold on;
plot(t, model, '--', 'LineWidth', 3);

grid on;
set(gca, 'XTick', [], 'YTick', []);

hold off;