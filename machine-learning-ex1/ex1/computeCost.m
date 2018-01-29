function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

Sum = 0.0;
h = 0.0;
m = size(y)(1);


for iter = 1:m
  
  h = theta(1)*X(iter,1) + theta(2)*X(iter,2); 
  Sum = Sum + (h - y(iter))^2;
  
end

J = (1/(2*m))*Sum;



% =========================================================================

end
