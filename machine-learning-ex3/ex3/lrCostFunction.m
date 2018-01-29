function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

%%%%%% Let's calculate J first:
    
  % Creating the vector h  
  h = zeros(size(X*theta));
  Ones = ones(size(y));
  
  % Calculating h, via the sigmoid function
  h = sigmoid(X * theta);
    
  % Calculating the sum
  Sum = (- y .* log(h) - (Ones - y) .* log(Ones - h));   
 
  thetaSquared = theta(2:end) .^ 2;
  
  thetaSum = sum(thetaSquared);
  
  % Finalizing the value of J
  J = (1/m) * sum(Sum) + (lambda/(2*m)) * thetaSum;

%%%%%% Now the gradient

  Sum2 = X' * (h - y); 
  Sum2 = (1/m) * Sum2;
  
  Sum2(2:end) = Sum2(2:end) + (lambda/m) * theta(2:end);    
    
  % The grad(j) is calculated by
  grad = Sum2;


% =============================================================

grad = grad(:);

end
