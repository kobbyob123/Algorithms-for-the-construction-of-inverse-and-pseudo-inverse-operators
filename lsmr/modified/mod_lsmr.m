function [x, errors, istop, itn, normr, normAr, normA, condA, normx]...
   = mod_lsmr(A, b, lambda, atol, btol, conlim, itnlim, localSize, show, x_groundtruth)
  
  % Initialize.
  if isa(A,'numeric')
    explicitA = true;
  elseif isa(A,'function_handle')
    explicitA = false;
  else
    error('SOL:lsmr:Atype','%s','A must be numeric or a function handle');
  end
    
  msg = ['The exact solution is  x = 0                              '
         'Ax - b is small enough, given atol, btol                  '
         'The least-squares solution is good enough, given atol     '
         'The estimate of cond(Abar) has exceeded conlim            '
         'Ax - b is small enough for this machine                   '
         'The least-squares solution is good enough for this machine'
         'Cond(Abar) seems to be too large for this machine         '
         'The iteration limit has been reached                      '];
  hdg1 = '   itn      x(1)       norm r    norm A''r';
  hdg2 = ' compatible   LS      norm A   cond A';
  pfreq  = 20;   % print frequency (for repeating the heading)
  pcount = 0;    % print counter
  % Determine dimensions m and n, and
  % form the first vectors u and v.
  % These satisfy  beta*u = b,  alpha*v = A'u.
  u    = b;
  beta = norm(u);
  if beta > 0
    u  = u/beta;
  end
  if explicitA
    v = A'*u;
    [m, n] = size(A);
  else  
    v = A(u,2);
    m = size(b,1);
    n = size(v,1);
  end
  
  minDim = min([m n]);
  
  % Set default parameters.
  if nargin < 3 || isempty(lambda)   , lambda    = 0;          end
  if nargin < 4 || isempty(atol)     , atol      = 1e-6;       end
  if nargin < 5 || isempty(btol)     , btol      = 1e-6;       end
  if nargin < 6 || isempty(conlim)   , conlim    = 1e+8;       end
  if nargin < 7 || isempty(itnlim)   , itnlim    = minDim;     end
  if nargin < 8 || isempty(localSize), localSize = 0;          end
  if nargin < 9 || isempty(show)     , show      = false;      end
  
  % -------------- Error Tracking Variables -------------------------
  errors_trk = ~isempty(x_groundtruth);
  errors     = zeros(itnlim, 1);   % pre-allocate; trimmed at the end
  % -------------- Error Tracking Variables -------------------------

  if show
    fprintf('\n\nLSMR-new            Least-squares solution of  Ax = b')
    fprintf('\nVersion 2.0                          15 Feb 2026')
    fprintf('\nThe matrix A has %8g rows  and %8g cols', m,n)
    fprintf('\nlambda = %16.10e', lambda )
    %fprintf('\natol   = %8.2e               conlim = %8.2e', atol,conlim)
    %fprintf('\nbtol   = %8.2e               itnlim = %8g'  , btol,itnlim)
  end
  
  alpha = norm(v);
  if alpha > 0
    v = (1/alpha)*v;
  end
  
  % Initialization for local reorthogonalization.
  localOrtho = false;
  if localSize > 0
    localPointer    = 0;
    localOrtho      = true;
    localVQueueFull = false;
    % Preallocate storage for the relevant number of latest v_k's.
    localV = zeros(n, min([localSize minDim]));
  end
  
  % Initialize variables for 1st iteration.
  itn      = 0;
  zetabar  = alpha*beta;
  alphabar = alpha;
  rho      = 1;
  rhobar   = 1;
  cbar     = 1;
  sbar     = 0;
  h    = v;
  hbar = zeros(n,1);
  x    = zeros(n,1);
  
  % Initialize variables for estimation of ||r||.
  betadd      = beta;
  betad       = 0;
  rhodold     = 1;
  tautildeold = 0;
  thetatilde  = 0;
  zeta        = 0;
  d           = 0;
  
  % Initialize variables for estimation of ||A|| and cond(A).
  normA2  = alpha^2;
  maxrbar = 0;
  minrbar = 1e+100;
  
  % Items for use in stopping rules.
  normb  = beta;
  istop  = 0;
  ctol   = 0;         if conlim > 0, ctol = 1/conlim; end
  normr  = beta;
  
  % Exit if b=0 or A'b = 0.
  normAr = alpha * beta;
  
  if normAr == 0, disp(msg(1,:)); return, end
  % Heading for iteration log.
  if show
    test1 = 1;
    test2 = alpha/beta;
    fprintf('\n\n%s%s'      , hdg1 , hdg2   )
    fprintf('\n%6g %12.5e'  , itn  , x(1)   )
    fprintf(' %10.3e %10.3e', normr, normAr )
    fprintf('  %8.1e %8.1e' , test1, test2  )
  end
  
  %------------------------------------------------------------------
  %     Main iteration loop.
  %------------------------------------------------------------------
  while itn < itnlim
    itn = itn + 1;
    % Perform the next step of the bidiagonalization to obtain the
    % next beta, u, alpha, v.  These satisfy the relations
    %      beta*u  =  A*v  - alpha*u,
    %      alpha*v  =  A'*u - beta*v.
    if explicitA
      u = A*v    - alpha*u;
    else 
      u = A(v,1) - alpha*u; 
    end
    beta = norm(u);
    if beta > 0
      u = (1/beta)*u;
      if localOrtho
        localVEnqueue(v);    % Store old v for local reorthogonalization of new v.
      end
      if explicitA
        v = A'*u   - beta*v;
      else
        v = A(u,2) - beta*v;
      end
      if localOrtho
        v = localVOrtho(v);  % Local-reorthogonalization of new v.
      end
      alpha  = norm(v);
      if alpha > 0,  v = (1/alpha)*v; end
    end
    
    % At this point, beta = beta_{k+1}, alpha = alpha_{k+1}.
    
    % Construct rotation Qhat_{k,2k+1}.
    alphahat = norm([alphabar lambda]);
    chat     = alphabar/alphahat;
    shat     = lambda/alphahat;
    % Use a plane rotation (Q_i) to turn B_i to R_i.
    rhoold   = rho;
    rho      = norm([alphahat beta]);
    c        = alphahat/rho;
    s        = beta/rho;
    thetanew = s*alpha;
    alphabar = c*alpha;
    % Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar.
    
    rhobarold = rhobar;
    zetaold   = zeta;
    thetabar  = sbar*rho;
    rhotemp   = cbar*rho;
    rhobar    = norm([cbar*rho thetanew]);
    cbar      = cbar*rho/rhobar;
    sbar      = thetanew/rhobar;
    zeta      =   cbar*zetabar;
    zetabar   = - sbar*zetabar;
    
    % Update h, h_hat, x.
    hbar      = h - (thetabar*rho/(rhoold*rhobarold))*hbar;
    x         = x + (zeta/(rho*rhobar))*hbar;
    h         = v - (thetanew/rho)*h;

    % ------------------------ Error Tracking -------------------------
    if errors_trk
        errors(itn) = norm(x - x_groundtruth);
    end
    % ------------------------ Error Tracking -------------------------

    % Estimate of ||r||.
    
    % Apply rotation Qhat_{k,2k+1}.
    betaacute =   chat* betadd;
    betacheck = - shat* betadd;
    
    % Apply rotation Q_{k,k+1}.
    betahat   =   c*betaacute;
    betadd    = - s*betaacute;
      
    % Apply rotation Qtilde_{k-1}.
    % betad = betad_{k-1} here.
    thetatildeold = thetatilde;
    rhotildeold   = norm([rhodold thetabar]);
    ctildeold     = rhodold/rhotildeold;
    stildeold     = thetabar/rhotildeold;
    thetatilde    = stildeold* rhobar;
    rhodold       =   ctildeold* rhobar;
    betad         = - stildeold*betad + ctildeold*betahat;
    % betad   = betad_k here.
    % rhodold = rhod_k  here.
    tautildeold   = (zetaold - thetatildeold*tautildeold)/rhotildeold;
    taud          = (zeta - thetatilde*tautildeold)/rhodold;
    d             = d + betacheck^2;
    normr         = sqrt(d + (betad - taud)^2 + betadd^2);
    
    % Estimate ||A||.
    normA2        = normA2 + beta^2;
    normA         = sqrt(normA2);
    normA2        = normA2 + alpha^2;
    
    % Estimate cond(A).
    maxrbar       = max(maxrbar,rhobarold);
    if itn>1 
      minrbar     = min(minrbar,rhobarold);
    end
    condA         = max(maxrbar,rhotemp)/min(minrbar,rhotemp);
    
    % Test for convergence.
    % Compute norms for convergence testing.
    normAr  = abs(zetabar);
    normx   = norm(x);
    
    % Now use these norms to estimate certain other quantities,
    % some of which will be small near a solution.
    test1   = normr /normb;
    test2   = normAr/(normA*normr);
    test3   =      1/condA;
    t1      =  test1/(1 + normA*normx/normb);
    rtol    = btol + atol*normA*normx/normb;
    
    % The following tests guard against extremely small values of
    % atol, btol or ctol.  (The user may have set any or all of
    % the parameters atol, btol, conlim  to 0.)
    % The effect is equivalent to the normAl tests using
    % atol = eps,  btol = eps,  conlim = 1/eps.
    % if itn >= itnlim,   istop = 7; end
    % if 1 + test3  <= 1, istop = 6; end
    % if 1 + test2  <= 1, istop = 5; end
    % if 1 + t1     <= 1, istop = 4; end
    
      % Allow for tolerances set by the user.
    if  test3 <= ctol,  istop = 3; end
    if  test2 <= atol,  istop = 2; end
    if  test1 <= rtol,  istop = 1; end
    
      % See if it is time to print something.
    if show
      prnt = 0;
      if n     <= 40       , prnt = 1; end
      if itn   <= 10       , prnt = 1; end
      if itn   >= itnlim-10, prnt = 1; end
      if rem(itn,10) == 0  , prnt = 1; end
      if test3 <= 1.1*ctol , prnt = 1; end
      if test2 <= 1.1*atol , prnt = 1; end
      if test1 <= 1.1*rtol , prnt = 1; end
      if istop ~=  0       , prnt = 1; end
      if prnt
	if pcount >= pfreq
	  pcount = 0;
          fprintf('\n\n%s%s'    , hdg1 , hdg2  )
	end
	pcount = pcount + 1;
        fprintf('\n%6g %12.5e'  , itn  , x(1)  )
        fprintf(' %10.3e %10.3e', normr, normAr)
        fprintf('  %8.1e %8.1e' , test1, test2 )
        fprintf(' %8.1e %8.1e'  , normA, condA )
      end
    end
    if istop > 0, break, end
  end % iteration loop
  
  % Print the stopping condition.
  if show
    fprintf('\n\nLSMR finished after %d iterations.\n', itn)
    fprintf('\n%s', msg(istop+1, :))
    fprintf('\nistop =%8g    normr =%8.1e'     , istop, normr )
    fprintf('    normA =%8.1e    normAr =%8.1e', normA, normAr)
    fprintf('\nitn   =%8g    condA =%8.1e'     , itn  , condA )
    fprintf('    normx =%8.1e\n', normx)
    % output residue
  end

  % ------------------ Ensuring Consistency -------------------------   
  if errors_trk
      errors = errors(1 : itn);
  end

  % end function lsmr

%---------------------------------------------------------------------
% Nested functions.
%---------------------------------------------------------------------
  function localVEnqueue(v)
  % Store v into the circular buffer localV.
    if localPointer < localSize
      localPointer = localPointer + 1;
    else
      localPointer = 1;
      localVQueueFull = true;
    end
    localV(:,localPointer) = v;
  end % nested function localVEnqueue
%---------------------------------------------------------------------
  function vOutput = localVOrtho(v)
  % Perform local reorthogonalization of V.
    vOutput = v;
    if localVQueueFull
      localOrthoLimit = localSize;
    else
      localOrthoLimit = localPointer;
    end
    for localOrthoCount = 1:localOrthoLimit
      vtemp   = localV(:, localOrthoCount);
      vOutput = vOutput - (vOutput'*vtemp)*vtemp;
    end
  end % nested function localVOrtho
end % function lsmr
