function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta); % number of thetas

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Let's calculate J first:

% Starting with the first sum, summing from 1 to m;
Sum = 0;
  for index = 1:m 
    
    % Calculating h, via the sigmoid function
    h(index) = sigmoid(theta'*X(index,:)');
    
    % Updating the sum
    Add = (- y(index) * log(h(index)) - (1 - y(index)) * log(1 - h(index)));
    Sum = Sum + Add;   
    
  end
  
% Now summing thetas
thetaSum = 0;
index = 0;

  for index = 2:n
    thetaSum = thetaSum + (theta(index))^2;
  end
  
% Finalizing the value of J
J = (((1/m) * Sum) + (lambda * (1/(2*m)) * thetaSum));


% Now the gradient
Sum = 0;
index = 0;
oldTheta = theta;

% Gradient for j = 0, in  this case, j = 1

for index = 1:m
    
    % Calculating h, via the sigmoid function
    h(index) = sigmoid(oldTheta'*X(index,:)');
    
    % Updating the sum
    Add = ((h(index) - y(index))*X(index,1));
    Sum = Sum + Add;    
end 
  
% The grad(j) is calculated by
grad(1) = (1/m) * Sum;

% For j >= 2
index = 0;
for j = 2:length(theta)
  
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
  grad(j) = (((1/m) * Sum) + (lambda * (1/m) * oldTheta(j)));
  
end








% =============================================================

end
