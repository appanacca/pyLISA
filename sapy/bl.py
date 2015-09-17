import numpy as np
import scipy.optimize as opt

def blasius(interactive):

	if nargin<1 
	   interactive = 1;
	end

	nb_pts = 100000;
	eta_max = 10;
	eta = np.linspace(0, eta_max, nb_pts);

	options = optimset('TolX', 1e-8, 'TolFun', 1e-8);
	s = fzero(@(s)Blasius_error(s, [0, eta_max]), 0.5);

	[eta, X] = Blasius_integration(s, eta);

	f  = X(:, 1);
	u  = X(:, 2);
	g  = X(:, 3);

	dg = -f.*g;

	% Diagnoses :

	eta_99  = interp1(u(abs(eta-3)<1), eta(abs(eta-3)<1), 0.99, 'spline');
	beta_1a = max(eta)-f(end);
	beta_1b = trapz(eta, 1-u);
	beta_2  = trapz(eta, u.*(1-u));
	beta_3  = trapz(eta, u.*(1-u.^2));
	H       = beta_1b/beta_2;
	
	return u,du,ddu
	"""
	if interactive

	  fprintf('\n----------------------------------------------------------------------\n');
	  fprintf(' BLASIUS BOUNDARY LAYER\n')
	  fprintf('----------------------------------------------------------------------\n\n');

	  fprintf(' Diagnoses (cf. Schlichting & Gersten, 8th edition, 2000) : \n\n');

	  fprintf(' Blasius parameters:       from Blasius equation f'''''' + f f'''' = 0\n\n');
	  fprintf('                       f''''(0) = %.4f (theory: 0.4696)\n', s);
	  fprintf('                       eta_99 = %.4f (theory: 3.472)\n', eta_99);
	  fprintf('   limit[eta-f(eta)] : beta_1 = %.4f (theory: 1.2168)\n', beta_1a);
	  fprintf('       int[1-u(eta)] : beta_1 = %.4f (theory: 1.2168)\n', beta_1b);
	  fprintf('        int[u*(1-u)] : beta_2 = %.4f (theory: 0.4696)\n', beta_2);
	  fprintf('      int[u*(1-u^2)] : beta_3 = %.4f (theory: 0.7385)\n', beta_3);

	  fprintf('\n Blasius thicknesses:       delta(x) = sqrt(x*nu/U_inf)\n\n');
	  fprintf('                     delta_99/delta(x)  = %.4f (theory: 4.910)\n', eta_99*sqrt(2));
	  fprintf('         displacement delta_1/delta(x)  = %.7f (theory: 1.7207876)\n', beta_1b*sqrt(2));
	  fprintf('             momentum delta_2/delta(x)  = %.4f (theory: 0.664)\n', beta_2*sqrt(2));
	  fprintf('               energy delta_3/delta(x)  = %.4f (theory: 1.0444)\n\n', beta_3*sqrt(2));
	  fprintf('         displacement delta_1/delta_99  = %.4f (theory: 0.34 ?)\n', beta_1b/eta_99);
	  fprintf('             momentum delta_2/delta_99  = %.4f (theory: 0.13 ?)\n', beta_2/eta_99);
	  fprintf('               energy delta_3/delta_99  = %.4f (theory: 0.20 ?)\n\n', beta_3/eta_99);
	  fprintf('          form/shape factor H = %.4f (theory: 2.59)\n\n', H);

	  figure
	  hold on
	  plot( u, eta, 'k.', 'Markersize', 2)
	  plot( g, eta, 'r.', 'Markersize', 2)
	  plot(dg, eta, 'b.', 'Markersize', 2)

	end """

	if nargout == 4

	  du  = g;
	  ddu = dg;

	  varargout(1) = {eta};
	  varargout(2) = {u};
	  varargout(3) = {du};
	  varargout(4) = {ddu};

	end


	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	function error = Blasius_error(s, eta_span)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	[eta, X] = Blasius_integration(s, eta_span);
	error = X(end, 2) - 1;

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	function [eta, X] = Blasius_integration(s, eta_span)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	X0 = [0, 0, s];

	warning off
	opts = odeset('RelTol', 1e-10, 'AbsTol', 1e-10, 'Jacobian', @Blasius_jacobian);
	[eta, X] = ode45(@(eta, X)Blasius_system(eta, X), eta_span, X0, opts);

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	function dX = Blasius_system(eta, X)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	f = X(1);
	u = X(2);
	g = X(3);

	dX = [u; g; -f*g];

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	function jac = Blasius_jacobian(eta, X)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	f = X(1);
	u = X(2);
	g = X(3);

	jac = [0, 1, 0; 0, 0, 1; -g, 0, -f];
