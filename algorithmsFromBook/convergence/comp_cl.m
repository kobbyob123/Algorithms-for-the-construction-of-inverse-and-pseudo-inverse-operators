% Full Comparison: F.2 vs F.3 vs LSMR
% Runs all three algorithms on the same problem and plots error curves.

clear; clc; close all;

% Problem Setup
N = 100; M = 60;
rng(42);
T      = randn(M, N);
x_true = randn(N, 1);
y      = T * x_true;
x_gt   = pinv(T) * y;   % minimum-norm least-squares solution

% Spectral Bounds (needed for F.2 and F.3 lambdas)
R  = T' * T;
ev = eig(R);
B  = max(ev);
A  = min(ev(ev > 1e-10));

lam_f2 = 2 / (A + B);
lam_f3 = sqrt(2 / (A^2 + B^2));

kappa = sqrt(B / A);
fprintf('Condition number κ(T) ≈ %.4f\n', kappa);
fprintf('λ_F2 = %.6f,  λ_F3 = %.6f\n\n', lam_f2, lam_f3);

% Run Algorithms
max_iter = 1000;

[err_f2,   ~] = f2(T, y, x_gt, lam_f2, max_iter);
[err_f3,   ~] = f3(T, y, x_gt, lam_f3, max_iter);

% lsmr with error tracking: (A, b, x_groundtruth, lambda, atol, btol, conlim, itnlim)
% atol/btol set very small so LSMR runs as many iterations as possible
[~, err_lsmr, istop, itn_lsmr] = lsmr_tracked(T, y, x_gt, 0, 1e-15, 1e-15, [], max_iter);

fprintf('LSMR stopped at iteration %d  (istop = %d)\n\n', itn_lsmr, istop);

% Theoretical Bounds for F.2 and F.3
r_f2    = (B - A) / (B + A);
r_f3    = (B^2 - A^2) / (B^2 + A^2);
iters   = (0:max_iter-1)';
bnd_f2  = (r_f2 .^ iters) * norm(x_gt);
bnd_f3  = (r_f3 .^ iters) * norm(x_gt);

% Plot
figure('Position', [100, 100, 1100, 500]);

% Left: all three errors together
subplot(1, 2, 1);
semilogy(err_f2,            'b-',  'LineWidth', 2,   'DisplayName', 'F.2 (Landweber, R)');
hold on;
semilogy(err_f3,            'g-',  'LineWidth', 2,   'DisplayName', 'F.3 (Landweber, R²)');
semilogy(err_lsmr,          'r--', 'LineWidth', 2.5, 'DisplayName', 'LSMR');
semilogy(bnd_f2,            'b:',  'LineWidth', 1,   'DisplayName', 'F.2 bound');
semilogy(bnd_f3,            'g:',  'LineWidth', 1,   'DisplayName', 'F.3 bound');
grid on;
legend('Location', 'northeast', 'FontSize', 9);
xlabel('Iteration');
ylabel('||x_k - x^+||');
title(sprintf('Error comparison  (κ ≈ %.1f)', kappa));
set(gca, 'FontSize', 10);
hold off;

% Right: zoom into LSMR range only (to see its curve clearly)
subplot(1, 2, 2);
semilogy(err_f2(1:itn_lsmr),   'b-',  'LineWidth', 2,   'DisplayName', 'F.2');
hold on;
semilogy(err_f3(1:itn_lsmr),   'g-',  'LineWidth', 2,   'DisplayName', 'F.3');
semilogy(err_lsmr,              'r--', 'LineWidth', 2.5, 'DisplayName', 'LSMR');
grid on;
legend('Location', 'northeast', 'FontSize', 9);
xlabel('Iteration');
ylabel('||x_k - x^+||');
title(sprintf('First %d iterations (LSMR range)', itn_lsmr));
set(gca, 'FontSize', 10);
hold off;

sgtitle(sprintf('F.2 vs F.3 vs LSMR  —  %d×%d system', M, N), 'FontSize', 14);

% Summary Table
fprintf('%-10s  %-12s  %-12s\n', 'Algorithm', 'Iterations', 'Final error');
fprintf('%-10s  %-12d  %-12.4e\n', 'F.2',  max_iter,  err_f2(end));
fprintf('%-10s  %-12d  %-12.4e\n', 'F.3',  max_iter,  err_f3(end));
fprintf('%-10s  %-12d  %-12.4e\n', 'LSMR', itn_lsmr,  err_lsmr(end));


%  LOCAL FUNCTION DEFINITIONS

function [x, errors, istop, itn, normr, normAr, normA, condA, normx] ...
   = lsmr_tracked(A, b, x_groundtruth, lambda, atol, btol, conlim, itnlim, localSize, show)

  if isa(A, 'numeric')
    explicitA = true;
  elseif isa(A, 'function_handle')
    explicitA = false;
  else
    error('lsmr_tracked:Atype', 'A must be a numeric matrix or a function handle.');
  end

  msg = ['The exact solution is  x = 0                              '
         'Ax - b is small enough, given atol, btol                  '
         'The least-squares solution is good enough, given atol     '
         'The estimate of cond(Abar) has exceeded conlim            '
         'Ax - b is small enough for this machine                   '
         'The least-squares solution is good enough for this machine'
         'Cond(Abar) seems to be too large for this machine         '
         'The iteration limit has been reached                      '];
  hdg1   = '   itn      x(1)       norm r    norm A''r';
  hdg2   = ' compatible   LS      norm A   cond A';
  pfreq  = 20;
  pcount = 0;

  u    = b;
  beta = norm(u);
  if beta > 0
    u = u / beta;
  end
  if explicitA
    v      = A' * u;
    [m, n] = size(A);
  else
    v = A(u, 2);
    m = size(b, 1);
    n = size(v, 1);
  end
  minDim = min(m, n);

  if nargin < 4  || isempty(lambda)   ,  lambda    = 0;      end
  if nargin < 5  || isempty(atol)     ,  atol      = 1e-6;   end
  if nargin < 6  || isempty(btol)     ,  btol      = 1e-6;   end
  if nargin < 7  || isempty(conlim)   ,  conlim    = 1e8;    end
  if nargin < 8  || isempty(itnlim)   ,  itnlim    = minDim; end
  if nargin < 9  || isempty(localSize),  localSize = 0;      end
  if nargin < 10 || isempty(show)     ,  show      = false;  end

  trackError = ~isempty(x_groundtruth);
  errors     = zeros(itnlim, 1);

  if show
    fprintf('\nLSMR_TRACKED: %g rows, %g cols, lambda=%.2e\n', m, n, lambda)
  end

  alpha = norm(v);
  if alpha > 0
    v = v / alpha;
  end

  localOrtho = false;
  if localSize > 0
    localPointer    = 0;
    localOrtho      = true;
    localVQueueFull = false;
    localV = zeros(n, min(localSize, minDim));
  end

  itn      = 0;
  zetabar  = alpha * beta;
  alphabar = alpha;
  rho      = 1;
  rhobar   = 1;
  cbar     = 1;
  sbar     = 0;
  h        = v;
  hbar     = zeros(n, 1);
  x        = zeros(n, 1);

  betadd      = beta;
  betad       = 0;
  rhodold     = 1;
  tautildeold = 0;
  thetatilde  = 0;
  zeta        = 0;
  d           = 0;

  normA2  = alpha^2;
  maxrbar = 0;
  minrbar = 1e+100;

  normb  = beta;
  istop  = 0;
  ctol   = 0;
  if conlim > 0
    ctol = 1 / conlim;
  end
  normr  = beta;
  normAr = alpha * beta;

  % FIX: initialise outputs before possible early return
  normA  = alpha;
  condA  = 1;
  normx  = 0;

  if normAr == 0
    disp(msg(1, :))
    errors = [];
    return
  end

  if show
    fprintf('\n%s%s\n', hdg1, hdg2)
  end

  while itn < itnlim

    itn = itn + 1;

    if explicitA
      u = A * v  - alpha * u;
    else
      u = A(v, 1) - alpha * u;
    end
    beta = norm(u);

    if beta > 0
      u = u / beta;
      if localOrtho,  localVEnqueue(v);  end
      if explicitA
        v = A' * u  - beta * v;
      else
        v = A(u, 2) - beta * v;
      end
      if localOrtho,  v = localVOrtho(v);  end
      alpha = norm(v);
      if alpha > 0,  v = v / alpha;  end
    end

    alphahat = norm([alphabar, lambda]);
    chat     = alphabar / alphahat;
    shat     = lambda   / alphahat;

    rhoold   = rho;
    rho      = norm([alphahat, beta]);
    c        = alphahat / rho;
    s        = beta     / rho;
    thetanew = s * alpha;
    alphabar = c * alpha;

    rhobarold = rhobar;
    zetaold   = zeta;
    thetabar  = sbar * rho;
    rhotemp   = cbar * rho;
    rhobar    = norm([cbar * rho, thetanew]);
    cbar      = cbar * rho / rhobar;
    sbar      = thetanew   / rhobar;
    zeta      =  cbar * zetabar;
    zetabar   = -sbar * zetabar;

    hbar = h - (thetabar * rho / (rhoold * rhobarold)) * hbar;
    x    = x + (zeta    / (rho  * rhobar))             * hbar;
    h    = v - (thetanew / rho)                        * h;

    % ── Error tracking (new) ──────────────────────────────────────────
    if trackError
      errors(itn) = norm(x - x_groundtruth);
    end

    betaacute =  chat * betadd;
    betacheck = -shat * betadd;
    betahat   =  c    * betaacute;
    betadd    = -s    * betaacute;

    thetatildeold = thetatilde;
    rhotildeold   = norm([rhodold, thetabar]);
    ctildeold     = rhodold  / rhotildeold;
    stildeold     = thetabar / rhotildeold;
    thetatilde    = stildeold * rhobar;
    rhodold       = ctildeold * rhobar;
    betad         = -stildeold * betad + ctildeold * betahat;
    tautildeold   = (zetaold - thetatildeold * tautildeold) / rhotildeold;
    taud          = (zeta    - thetatilde    * tautildeold) / rhodold;
    d             = d + betacheck^2;
    normr         = sqrt(d + (betad - taud)^2 + betadd^2);

    normA2 = normA2 + beta^2;
    normA  = sqrt(normA2);
    normA2 = normA2 + alpha^2;

    maxrbar = max(maxrbar, rhobarold);
    if itn > 1
      minrbar = min(minrbar, rhobarold);
    end
    condA = max(maxrbar, rhotemp) / min(minrbar, rhotemp);

    normAr = abs(zetabar);
    normx  = norm(x);

    test1 = normr  / normb;
    test2 = normAr / (normA * normr);
    test3 = 1      / condA;
    t1    = test1  / (1 + normA * normx / normb);
    rtol  = btol   + atol * normA * normx / normb;

    if itn   >= itnlim    ,  istop = 7;  end
    if 1 + test3 <= 1     ,  istop = 6;  end
    if 1 + test2 <= 1     ,  istop = 5;  end
    if 1 + t1    <= 1     ,  istop = 4;  end
    if test3 <= ctol       ,  istop = 3;  end
    if test2 <= atol       ,  istop = 2;  end
    if test1 <= rtol       ,  istop = 1;  end

    if show
      prnt = false;
      if n <= 40 || itn <= 10 || itn >= itnlim-10 || mod(itn,10)==0
        prnt = true;
      end
      if test3 <= 1.1*ctol || test2 <= 1.1*atol || test1 <= 1.1*rtol
        prnt = true;
      end
      if istop ~= 0,  prnt = true;  end
      if prnt
        if pcount >= pfreq
          pcount = 0;
          fprintf('\n%s%s\n', hdg1, hdg2)
        end
        pcount = pcount + 1;
        fprintf('%6g %12.5e %10.3e %10.3e  %8.1e %8.1e %8.1e %8.1e\n', ...
                itn, x(1), normr, normAr, test1, test2, normA, condA)
      end
    end

    if istop > 0,  break;  end

  end

  errors = errors(1:itn);

  if show
    fprintf('\nLSMR finished: itn=%d, istop=%d\n%s\n', itn, istop, msg(istop+1,:))
  end

  % ── Nested helpers ───────────────────────────────────────────────────
  function localVEnqueue(v_in)
    if localPointer < localSize
      localPointer = localPointer + 1;
    else
      localPointer    = 1;
      localVQueueFull = true;
    end
    localV(:, localPointer) = v_in;
  end

  function vOut = localVOrtho(v_in)
    vOut = v_in;
    if localVQueueFull,  limit = localSize;
    else,                limit = localPointer;
    end
    for k = 1:limit
      vt   = localV(:, k);
      vOut = vOut - (vOut' * vt) * vt;
    end
  end

end % lsmr_tracked
