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

Y = full(sparse (1:rows (y), y, 1) ); % m,s3

A1 = [ones(m, 1) X]; %  m,s1 + 1

Z2 = Theta1 * A1'; % s2,m

A2 = [ones(m, 1) sigmoid(Z2')]; % m,s2 + 1

Z3 = Theta2 * A2'; % s3,m

A3 = sigmoid(Z3'); % m,s3

G = A3 ;
LL = log(G); % m,s3
LR = log(1 - G); % m,s3
 % Tg = theta;
 % Tg(1, 1) = 0;
 % R = (Tg' * Tg) * lambda / (2 * m);

for i = 1:m
  % calculate i-th component of the Cost
  LLi = LL(i, :); % 1,k
  LRi = LR(i, :); % 1,k
  Yi =  Y(i, :) ; % 1,k
  J = J + (LLi * Yi' + LRi * (1 - Yi')) ;  

  % calculate backprop   ( Theta1 size s2,s1+1    Theta2 size k,s2+1 )
  A3i = A3(i, :); % 1,k
  d3 = A3i - Yi; % 1,k
  A2i = A2(i, :); % 1,s2+1
  
  gA2i = A2i' .* (1-A2i'); % s2 + 1, 1  ------> derivative of A2i
  d2 = (Theta2' * d3') .* gA2i;   % s2 + 1,1
  d2 = d2(2:end) ; % s2,1
  %d2(1) = 0;
  A1i = A1(i, :); % 1,s1+1

  Theta1_grad = Theta1_grad + d2 * A1i;
  Theta2_grad = Theta2_grad + d3' * A2i;    
end

% zero bias column
T1 = [zeros(size(Theta1, 1),1) Theta1(:, 2:end)];
T2 = [zeros(size(Theta2, 1),1) Theta2(:, 2:end)];


Theta1_grad = Theta1_grad ./ m + T1.*(lambda/m);
Theta2_grad = Theta2_grad ./ m + T2.*(lambda/m);


% calculate regularization
  R = (sumsq(T1(:)) + sumsq(T2(:))) * lambda / (2 * m);

  J = - (J / m) + R;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
