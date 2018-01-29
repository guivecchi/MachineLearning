function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% PART 1

% Let's calculate h(x) first of all
copyX = X;

h = zeros(size(copyX, 1), 1);
  % Add ones to the X data matrix
  copyX = [ones(m, 1) copyX];

  % Let's first calculate the hidden layer
  hidden = sigmoid(copyX * Theta1');

  % Add ones to the hidden data matrix
  hidden = [ones(m, 1) hidden];

  % Calculating the output layer
  h = sigmoid(hidden * Theta2');

% h is 5000 x 10 matrix; y is 5000 x 1; i goes from 1 to 5000;
Sum = 0;

for i = 1:m
    newY = zeros(num_labels, 1);
    value = y(i);
    newY(value) = 1;
    
    for k = 1:num_labels
      Sum = Sum + (- newY(k) .* log(h(i,k)) - (1 - newY(k)) .* log(1 - h(i,k)));
    end
end

% if lambda == 0
  J = (1 / m) * Sum;


% if lambda == 1
  sumReg = 0;
  % Regularization
  sumReg = sum(sum(Theta1(:, 2:size(Theta1, 2)) .^ 2)) + sum(sum(Theta2(:, 2:size(Theta2, 2)) .^ 2));

  % Cost Function
  J = (1 / m) * Sum + (lambda / (2 * m)) * sumReg; 
  
% PART 2

bigDelta_1 = zeros(size(Theta1));
bigDelta_2 = zeros(size(Theta2));

for t = 1:m
  % Creating the y we're using
  newY = zeros(num_labels, 1);
  value = y(t);
  newY(value) = 1;
    
  % Step 1
  a_1 = zeros(size(X, 2), 1);
  a_1 = X(t, :)';
  
  % Feed forward
  
    % Add ones to a_1
    a_1 = [1; a_1];

    % Let's first calculate the hidden layer
    z_2 = Theta1 * a_1;
    a_2 = sigmoid(z_2);

    % Add ones to a_2
    a_2 = [1; a_2];

    % Calculating a_3
    z_3 = Theta2 * a_2;
    a_3 = sigmoid(z_3);
    
    % Step 2
    delta_3 = zeros(size(a_3));
    delta_3 = a_3 - newY;
    
    % Step 3
    derivativeGz2 = sigmoidGradient(z_2);
    derivativeGz2 = [1; derivativeGz2];
    
    delta_2 = Theta2' * delta_3 .* derivativeGz2;

    % Step 4
    delta_2 = delta_2(2:end);
    
    bigDelta_1 = bigDelta_1 + delta_2 * a_1';    
    bigDelta_2 = bigDelta_2 + delta_3 * a_2';
    
  
end  

Theta1_grad = (1 / m) .* bigDelta_1; 
Theta2_grad = (1 / m) .* bigDelta_2;

% Regularization
% Theta_1
for i = 1:size(Theta1, 1)
  for j = 2:size(Theta1, 2)
    Theta1_grad(i, j) = Theta1_grad(i, j) + (lambda / m) * Theta1(i, j);
  end
end

for i = 1:size(Theta2, 1)
  for j = 2:size(Theta2, 2)
    Theta2_grad(i, j) = Theta2_grad(i, j) + (lambda / m) * Theta2(i, j);
  end
end


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
