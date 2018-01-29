function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

% Let's calculate J first:

% Starting with the sum, summing from 1 to m;
Sum = 0;
  for index = 1:m 
    
    % Calculating h, via the sigmoid function
    h(index) = sigmoid(theta'*X(index,:)');
    
    % Updating the sum
    Add = (- y(index) * log(h(index)) - (1 - y(index)) * log(1 - h(index)));
    Sum = Sum + Add;   
    
  end
  
  % Finalizing the value of J
  J = (1/m) * Sum;

% Now the gradient

% Start creating a loop to calculate each theta(j) on the vector
index = 0;

oldTheta = theta;
for j = 1:length(theta)
  
  Sum = 0;
  % Calculating the sum
  for index = 1:m
    
    % Calculating h, via the sigmoid function
    h(index) = sigmoid(oldTheta'*X(index,:)');
    
    % Updating the sum
    Add = ((h(index) - y(index))*X(index,j));
    Sum = Sum + Add;    
  end 
  
  % The grad(j) is calculated by
  grad(j) = (1/m) * Sum;
  
end






% =============================================================

end
