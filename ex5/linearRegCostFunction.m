function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
hx = X*theta;
J = sum((hx-y).*(hx-y));
t_theta= theta(2:end);
J = J+lambda*sum(t_theta.*t_theta);
J = J/2/m;

grad = (hx-y).*X;
t_theta = theta;
t_theta(1) = 0;
grad = sum(grad);
grad = grad' + lambda*t_theta;
grad = grad / m;
%disp(grad)


%disp(size(X))
%disp(size(theta))











% =========================================================================

grad = grad(:);

end
