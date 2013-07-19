function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
temp = zeros(3,1);
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
u1 = 0;
u2 = 0;
u3 = 0;	
	for i=1:m,
		u1 = u1 + ((X(i,:)*theta) - y(i))*X(i,1);
		u2 = u2 + ((X(i,:)*theta) - y(i))*X(i,2);
		u3 = u3 + ((X(i,:)*theta) - y(i))*X(i,3);

		end
		temp(1) = theta(1) - alpha * (1/m) * u1; 

		temp(2) = theta(2) - alpha * (1/m) * u2; 
		temp(3) = theta(3) - alpha * (1/m) * u3; 
	theta = temp;










    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
