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
a = zeros(size(y));
a = ((X * theta)-y) .^ 2;

r = zeros(size(theta));
r = theta .^ 2;
r(1) = 0;

J = sum(a(:))/(2*m) + (lambda/(2*m))*sum(r(:));

temp = zeros(size(y));

reg = zeros(size(theta));
for j=1:length(theta),
	reg(j) = (lambda/m)*theta(j);

reg(1) = 0;
for i=1:length(theta),
	temp = ((X * theta)-y) .* X(:,i);
	grad(i) = sum(temp(:))/m + reg(i);
end




% =========================================================================

grad = grad(:);

end
