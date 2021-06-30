function [q, k] = mmq_exp_given_tol(a,i,tol)
%MMQ_BOUNDS_EXP_GAUSS computation of lower and upper bounds of the element (i,i) of the exponential
% of a symmetric matrix a using Lanczos
% kmax iterations
%
% lmin and lmax are the lower and upper bounds for the smallest and largest eigenvalues
%
%  bg :         Gauss
%  bgrl, bgru : Gauss-Radau
%  bgl :        Gauss-Lobatto
%
% we compute the exponential of the Jacobi matrix. This can be costly
%  
% Author G. Meurant
% March 2008
%

kmax = size(a, 1);

jj=sparse(kmax,kmax);
n=size(a,1);
ei=zeros(n,1);
ei(i)=1;
x1=zeros(n,1);
x=ei;
gam=0;
ax=a*x;
om=x'*ax;
jj(1,1)=om;
q=exp(om);
r=ax-om*x;
gam2=r'*r;
gam=sqrt(gam2);
if gam == 0
    k = 1;
   return
end
x1=x;
x=r/gam;

% Lanczos iterations
prec = q;
if kmax > 1
 for k=2:kmax
  gam1=gam;
  gam21=gam2;
  ax=a*x;
  om=x'*ax;
  jj(k,k)=om;
  jj(k,k-1)=gam;
  jj(k-1,k)=gam;
  r=ax-om*x-gam*x1;
  gam2=r'*r;
  gam=sqrt(gam2);
  x1=x;

  % Gauss
  ejj=expm(full(jj(1:k,1:k)));
  q = ejj(1,1);
  
  % check for breakdown
  if gam == 0
   break
  end
  x=r/gam;
  if (abs(q- prec)/abs(q) < tol)
      break;
  else
      prec = q;
  end
 end
end
