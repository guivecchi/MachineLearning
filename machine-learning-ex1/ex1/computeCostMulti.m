function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

Sum = 0.0;
#{ h = 0.0;
   m = size(y)(1);


   for iter = 1:m
     h = 0.0;  
     for i = 1:size(X)(2)
       h = h + theta(i)*X(iter,i); 
     end
 
     Sum = Sum + (h - y(iter))^2;
  
   end
#}

Sum = ((X*theta - y)'*(X*theta - y));


J = ((1/(2*m))*Sum);




% =========================================================================

end
