function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

u = 0;
v = 0;
w=0;
for i=1:m,
	v=v + [(-y(i)*log(sigmoid(theta' * X(i,:)')))-((1-y(i))*log(1 - sigmoid(theta'*X(i,:)')))];
end
for k=2:length(theta),
	w = w + theta(k)*theta(k);
end
J = (1/m * v) + lambda/(2*m) * w;


for i=1:length(X),
	u = u + (sigmoid(theta' * X(i,:)')-y(i)) * X(i,1);
end
grad(1) = 1/m * u;

for j=2:length(theta),
	u = 0;
	for i=1:length(X),
		u = u + (sigmoid(theta' * X(i,:)')-y(i)) * X(i,j);
	end
	grad(j) = (1/m * u) + (lambda/m * theta(j));

end


% =============================================================

end
