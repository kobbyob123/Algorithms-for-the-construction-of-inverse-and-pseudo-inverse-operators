function [x, errors, istop, itn, normr, normAr, normA, condA, normx] ...
   = lsmr_2(A, b, x_groundtruth, lambda, atol, btol, conlim, itnlim, localSize, show)
% LSMR_TRACKED  LSMR solver with per-iteration error tracking against a ground truth.
%
%   [X, ERRORS] = LSMR_TRACKED(A, B, X_GROUNDTRUTH) solves the least-squares
%   problem min ||b - Ax||_2 and also returns ERRORS(k) = ||x_k - x_groundtruth||
%   at each iteration, enabling direct comparison with f2/f3 algorithms.
%
%   X_GROUNDTRUTH is the reference solution (e.g. pinv(A)*b). Pass [] to skip
%   error tracking (ERRORS will be empty).
%
%   All remaining parameters and outputs are identical to the original LSMR:
%
%   [X, ERRORS, ISTOP, ITN, NORMR, NORMAR, NORMA, CONDA, NORMX] = LSMR_TRACKED(...)
%
%   A may be a numeric matrix OR a function handle AFUN where:
%       AFUN(v, 1)  returns  A*v   (forward operator)
%       AFUN(u, 2)  returns  A'*u  (adjoint operator)
%
%   LAMBDA  - Tikhonov regularisation parameter (default 0, i.e. no regularisation)
%   ATOL    - Tolerance on ||A'r|| / (||A|| ||r||)   (default 1e-6)
%   BTOL    - Tolerance on ||r|| / ||b||              (default 1e-6)
%   CONLIM  - Terminate if cond(A) exceeds CONLIM     (default 1e8)
%   ITNLIM  - Maximum number of iterations            (default min(m,n))
%   LOCALSIZE - Number of v-vectors for reorthogonalisation (0 = none, Inf = full)
%   SHOW    - Print iteration log if true             (default false)
%
%   ISTOP codes:
%      0 = x=0 is a solution
%      1 = ||r||   is small enough (atol, btol satisfied)
%      2 = ||A'r|| is small enough (atol satisfied for normal equation)
%      3 = cond(A) > CONLIM
%      4,5,6 = same as 1,2,3 but limited by machine precision
%      7 = iteration limit reached
%
% -------------------------------------------------------------------------
% Original LSMR by D. C.-L. Fong & M. A. Saunders (Stanford, 2010).
% Error-tracking wrapper and MATLAB fixes by thesis student, 2026.
% Reference: Fong & Saunders, SIAM J. Sci. Comput., 2011.
% -------------------------------------------------------------------------

  % ---- Validate A --------------------------------------------------------
  if isa(A, 'numeric')
    explicitA = true;
  elseif isa(A, 'function_handle')
    explicitA = false;
  else
    error('lsmr_tracked:Atype', 'A must be a numeric matrix or a function handle.');
  end

  % ---- Status messages ---------------------------------------------------
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

  % ---- First Golub-Kahan vectors: beta*u = b,  alpha*v = A'*u -----------
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

  % ---- Default parameters ------------------------------------------------
  if nargin < 4  || isempty(lambda)   ,  lambda    = 0;      end
  if nargin < 5  || isempty(atol)     ,  atol      = 1e-6;   end
  if nargin < 6  || isempty(btol)     ,  btol      = 1e-6;   end
  if nargin < 7  || isempty(conlim)   ,  conlim    = 1e8;    end
  if nargin < 8  || isempty(itnlim)   ,  itnlim    = minDim; end
  if nargin < 9  || isempty(localSize),  localSize = 0;      end
  if nargin < 10 || isempty(show)     ,  show      = false;  end

  trackError = ~isempty(x_groundtruth);
  errors     = zeros(itnlim, 1);   % pre-allocate; trimmed at the end

  if show
    fprintf('\n\nLSMR_TRACKED    Least-squares solution of  Ax = b')
    fprintf('\nThe matrix A has %8g rows  and %8g cols', m, n)
    fprintf('\nlambda = %16.10e', lambda)
    fprintf('\natol   = %8.2e               conlim = %8.2e', atol, conlim)
    fprintf('\nbtol   = %8.2e               itnlim = %8g\n', btol, itnlim)
  end

  alpha = norm(v);
  if alpha > 0
    v = v / alpha;
  end

  % ---- Local reorthogonalisation setup -----------------------------------
  localOrtho = false;
  if localSize > 0
    localPointer    = 0;
    localOrtho      = true;
    localVQueueFull = false;
    localV = zeros(n, min(localSize, minDim));
  end

  % ---- Initialise iteration variables ------------------------------------
  itn      = 0;
  zetabar  = alpha * beta;
  alphabar = alpha;
  rho      = 1;
  rhobar   = 1;
  cbar     = 1;
  sbar     = 0;

  h    = v;
  hbar = zeros(n, 1);
  x    = zeros(n, 1);

  % Variables for ||r|| estimate
  betadd      = beta;
  betad       = 0;
  rhodold     = 1;
  tautildeold = 0;
  thetatilde  = 0;
  zeta        = 0;
  d           = 0;

  % Variables for ||A|| and cond(A) estimates
  normA2  = alpha^2;
  maxrbar = 0;
  minrbar = 1e+100;

  % Stopping rule variables
  normb  = beta;
  istop  = 0;
  ctol   = 0;
  if conlim > 0
    ctol = 1 / conlim;   % FIX: was "end;" — removed spurious semicolon after end
  end
  normr  = beta;
  normAr = alpha * beta;

  % FIX: early-exit when b=0 or A'b=0 now initialises ALL outputs to avoid
  %      MATLAB warnings about uninitialised variables.
  if normAr == 0
    disp(msg(1, :))
    normA  = alpha;   % best estimate available
    condA  = 1;
    normx  = 0;
    errors = [];
    return
  end

  if show
    test1 = 1;
    test2 = alpha / beta;
    fprintf('\n%s%s\n', hdg1, hdg2)
    fprintf('%6g %12.5e %10.3e %10.3e  %8.1e %8.1e\n', ...
            itn, x(1), normr, normAr, test1, test2)
  end

  % ========================================================================
  %  Main iteration loop
  % ========================================================================
  while itn < itnlim

    itn = itn + 1;

    % ---- Golub-Kahan bidiagonalization step --------------------------------
    %   beta_{k+1} * u_{k+1} = A * v_k  -  alpha_k * u_k
    %   alpha_{k+1} * v_{k+1} = A'* u_{k+1}  -  beta_{k+1} * v_k
    if explicitA
      u = A * v  - alpha * u;
    else
      u = A(v, 1) - alpha * u;
    end
    beta = norm(u);

    if beta > 0
      u = u / beta;
      if localOrtho
        localVEnqueue(v);
      end
      if explicitA
        v = A' * u  - beta * v;
      else
        v = A(u, 2) - beta * v;
      end
      if localOrtho
        v = localVOrtho(v);
      end
      alpha = norm(v);
      if alpha > 0
        v = v / alpha;
      end
    end

    % ---- Plane rotations ---------------------------------------------------
    % Qhat_{k,2k+1}: incorporate regularisation parameter lambda
    alphahat = norm([alphabar, lambda]);
    chat     = alphabar / alphahat;
    shat     = lambda   / alphahat;

    % Q_i: turn B_i to R_i
    rhoold   = rho;
    rho      = norm([alphahat, beta]);
    c        = alphahat / rho;
    s        = beta     / rho;
    thetanew = s * alpha;
    alphabar = c * alpha;

    % Qbar_i: turn R_i^T to R_i^bar
    rhobarold = rhobar;
    zetaold   = zeta;
    thetabar  = sbar * rho;
    rhotemp   = cbar * rho;
    rhobar    = norm([cbar * rho, thetanew]);
    cbar      = cbar * rho / rhobar;
    sbar      = thetanew   / rhobar;
    zeta      =  cbar * zetabar;
    zetabar   = -sbar * zetabar;

    % ---- Update h, hbar, x -------------------------------------------------
    hbar = h - (thetabar * rho / (rhoold * rhobarold)) * hbar;
    x    = x + (zeta    / (rho  * rhobar))             * hbar;
    h    = v - (thetanew / rho)                        * h;

    % ---- Track error against ground truth ----------------------------------
    % This is the key addition for comparison with f2/f3.
    % x has been updated above and represents the current best estimate.
    if trackError
      errors(itn) = norm(x - x_groundtruth);
    end

    % ---- Estimate ||r|| via Qtilde_{k-1} rotation --------------------------
    thetatildeold = thetatilde;
    rhotildeold   = norm([rhodold, thetabar]);
    ctildeold     = rhodold  / rhotildeold;
    stildeold     = thetabar / rhotildeold;
    thetatilde    = stildeold * rhobar;
    rhodold       = ctildeold * rhobar;
    betad         = -stildeold * betad + ctildeold * betahat;
    tautildeold   = (zetaold - thetatildeold * tautildeold) / rhotildeold;
    taud          = (zeta    - thetatilde    * tautildeold) / rhodold;

    % Apply Qhat and Q rotations for the ||r|| estimate components
    betaacute =  chat * betadd;
    betacheck = -shat * betadd;
    betahat   =  c    * betaacute;
    betadd    = -s    * betaacute;

    d     = d + betacheck^2;
    normr = sqrt(d + (betad - taud)^2 + betadd^2);

    % ---- Estimate ||A|| and cond(A) ----------------------------------------
    normA2 = normA2 + beta^2;
    normA  = sqrt(normA2);
    normA2 = normA2 + alpha^2;

    maxrbar = max(maxrbar, rhobarold);
    if itn > 1
      minrbar = min(minrbar, rhobarold);
    end
    condA = max(maxrbar, rhotemp) / min(minrbar, rhotemp);

    % ---- Convergence tests -------------------------------------------------
    normAr = abs(zetabar);
    normx  = norm(x);

    test1 = normr  / normb;
    test2 = normAr / (normA * normr);
    test3 = 1      / condA;
    t1    = test1  / (1 + normA * normx / normb);
    rtol  = btol   + atol * normA * normx / normb;

    % Machine-precision guards
    if itn   >= itnlim    ,  istop = 7;  end
    if 1 + test3 <= 1     ,  istop = 6;  end
    if 1 + test2 <= 1     ,  istop = 5;  end
    if 1 + t1    <= 1     ,  istop = 4;  end
    % User-tolerance tests
    if test3 <= ctol       ,  istop = 3;  end
    if test2 <= atol       ,  istop = 2;  end
    if test1 <= rtol       ,  istop = 1;  end

    % ---- Optional iteration log --------------------------------------------
    if show
      prnt = false;
      if n        <= 40        ,  prnt = true;  end
      if itn      <= 10        ,  prnt = true;  end
      if itn      >= itnlim-10 ,  prnt = true;  end
      if mod(itn, 10) == 0     ,  prnt = true;  end
      if test3 <= 1.1 * ctol   ,  prnt = true;  end
      if test2 <= 1.1 * atol   ,  prnt = true;  end
      if test1 <= 1.1 * rtol   ,  prnt = true;  end
      if istop ~= 0            ,  prnt = true;  end
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

    if istop > 0
      break
    end

  end  % main iteration loop

  % ---- Trim error vector to actual number of iterations performed --------
  errors = errors(1:itn);

  % ---- Final log ---------------------------------------------------------
  if show
    fprintf('\nLSMR_TRACKED finished after %d iterations.\n', itn)
    fprintf('%s\n', msg(istop + 1, :))
    fprintf('istop = %d    normr = %8.1e    normA = %8.1e    normAr = %8.1e\n', ...
            istop, normr, normA, normAr)
    fprintf('itn   = %d    condA = %8.1e    normx = %8.1e\n', itn, condA, normx)
  end

% =========================================================================
%  Nested helper functions for local reorthogonalisation
% =========================================================================

  function localVEnqueue(v_in)
  % Store v_in into the circular buffer localV.
    if localPointer < localSize
      localPointer = localPointer + 1;
    else
      localPointer     = 1;
      localVQueueFull  = true;
    end
    localV(:, localPointer) = v_in;
  end

  function vOut = localVOrtho(v_in)
  % Gram-Schmidt reorthogonalisation against the stored v-vectors.
    vOut = v_in;
    if localVQueueFull
      limit = localSize;
    else
      limit = localPointer;
    end
    for k = 1:limit
      vt   = localV(:, k);
      vOut = vOut - (vOut' * vt) * vt;
    end
  end

end  % function lsmr_2
