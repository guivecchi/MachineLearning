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

% First term
Sum = 0.0;
Sum = ((X*theta - y)'*(X*theta - y));

% Second term
thetaSum = 0.0;
thetaSquared = theta(2:end) .^ 2;
thetaSum = sum(thetaSquared);

% Cost function
J = ((1 / (2 * m)) * Sum) + ((lambda / (2 * m)) * thetaSum);

% Let's calculate the gradient
for j = 1:size(theta)  
  Sum = 0.0;
  for i = 1:m
    h = 0.0;  
      for k = 1:size(X)(2)
        h = h + theta(k)*X(i,k); 
      end
 
    Sum = Sum + (h - y(i))*X(i,j);
  end
  grad(j) = (1 / m) * Sum;
end

for j = 2:size(theta)
  grad(j) = grad(j) + ((lambda / m) * theta(j));
end


% =========================================================================

grad = grad(:);

end
