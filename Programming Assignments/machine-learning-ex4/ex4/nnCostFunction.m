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

% reshaping the Theta1,Theta2 into weight matrices of 25x401 and 10x26
% Rolled vector elements 10025 + 260 = 10285

% Theta1 = reshape(nn_params(1:25*401),25, 401)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);  % 5000 
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1)); % 25 x 401 
Theta2_grad = zeros(size(Theta2)); % 10 x 26 

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




% Part1 : Implement the cost function for the neural network 

% create the z_2 for layer 2 / hidden layer 
X = [ones(m, 1) X];   % adding the bias colums to the X matrix - 5000 x 401
a_1 = X;

z_2 = a_1 * Theta1'; % 5000 x 401 * 401 x 25
a_2 = sigmoid(z_2); % 5000 x 25 

% create the z_3 for the output layer / layer3 
% add a bias unit to the a_2 
a_2 = [ones(size(a_2,1),1) a_2]; % 5000 x 26

z_3 = a_2 * Theta2'; % 5000 x 26 * 26 x 10 
a_3 = sigmoid(z_3); % 5000 x 10 = prediction of the network hthetax @ output layer

% converting the y matrix into individual label matrices

y_matrix = eye(num_labels)(y,:) ; % results in a individual vector for each label

% Compute the cost now 
% make sure to use the double sum since there are 2 summations in this cost function 
% of neural networks 


% a_3 = 5000 x 10 , y_matrix = 5000 x 10 
% vectorised operation 

J_0 = (-1/m)* (log(a_3)' * y_matrix + log(1-a_3)' * (1-y_matrix));
J = trace(J_0);  
% sums up the diagonal elements resulting in the cost function J 

% Lets add the regularised term to the cost function 
% Theta1 - 25 x 401 , Theta2 - 10 x 26 

% Note : Regularisation is not needed for the bias column in the a_3 or whatever
% 1st col in theta would be multiplied with the bias cols of 1 in X 
% Hence making the first column of theta to zero 

% check the ex4 pdf file and refer the math concept below this file for understanding 

Theta1(:,1) = 0 ; % 25 x 400
Theta2(:,1) = 0;  %  10 x 25 

reg_theta1 = Theta1' * Theta1;
reg_theta2 = Theta2' * Theta2;

reg1 = trace(reg_theta1);
reg2 = trace(reg_theta2);

J_reg = (lambda/(2*m))*(reg1 + reg2);

J = J + J_reg;


##for j= 1: size(Theta1,1) % row size 25
##  for k = 2: size(Theta1,2) % col size 401-1 
##    % J_reg1 = sum(sum(Theta1(j,k).^2));
##    J_reg_k1 = sum(Theta1(j,k).^2);
##  endfor
##  J_reg_i1 = sum(J_reg_k1);
##endfor
##
##for j= 1: size(Theta2,1) % row size 10
##  for k = 2: size(Theta2,2) % col size 26-1 
##    J_reg_k2 = sum(Theta2(j,k).^2);
##  endfor
##  J_reg_i2 = sum(J_reg_k2);
##endfor
##
##J_reg = (lambda/(2*m))*(J_reg_i1+J_reg_i2);
##
##J = J + J_reg;


% using for loop didnt work - wrong nesting ???

##for i = 1:m
##  for k = 1:num_labels
##    J_k_i = sum(sum(-y_matrix(i,k).* log(a_3(i,k)) - (1-y_matrix(i,k)).* log(1-a_3(i,k))));
##  endfor
##endfor
##J = (1/m) * J_k_i;

% Look this resource section in the course why the multiplication is done this way
% https://www.coursera.org/learn/machine-learning/discussions/all/threads/AzIrrO7wEeaV3gonaJwAFA

% Math behind the above computation of J

##A = \begin{bmatrix} a & b\\ c & d\\ e & f \end{bmatrix} , B = \begin{bmatrix} m & n\\ o & p\\ q & r \end{bmatrix}
##The sum over the rows and columns of the element-wise product is:
##
##??A.*B=am+bn+co+dp+eq+fr


##Now let's detail the math for this using a matrix product. Since A and B are the same size, 
##but the number of rows and columns are not the same, 
##we must transpose one of the matrices before we compute the product. 
##Let's transpose the 'A' matrix, so the product matrix will be size (K x K). 
##We could of course invert the 'B' matrix, but then the product matrix would be size (m x m). 
##The (m x m) matrix is probably a lot larger than (K x K).
##
##It turns out (and is left for the reader to prove) that both the (m x m) and (K x K) 
##matrices will give the same results for the cost J.

##{A}'* B= \begin{bmatrix}a & c & e\\b & d & f\end{bmatrix}*\begin{bmatrix}m & n\\o & p\\q & r\end{bmatrix}
##After the matrix product, we get:
##
##{A}'*B = \begin{bmatrix}(am + co + eq) & (an + cp + er)\\(bm + do + fq) & (bn + dp + fr)\end{bmatrix}

##The next step is to compute the sum of the diagonal elements using the "trace()" command, 
##or by sum(sum(...)) 
##after element-wise multiplying by an identity matrix of size (K x K).
##


# Part2 - Implement Backpropagation Algorithm 

# 2.1  - Implement sigmoid gradient function in Sigmoid gradient function file 

# Initialise the weights using epsilon_init 

# Implement the backpropagation algorithm method 


d3 = a_3 - y_matrix; % 5000 x 10 

% need the return value of sigmoid gradient return value for computing d2 
gdash_z_2 = sigmoidGradient(z_2); % 5000 x 25 
d2 = (d3*Theta2(:,2:end)).* gdash_z_2; % 5000 x 10 * 10 x 25 

##Note: Excluding the first column of Theta2 is because the hidden layer bias unit has no connection to the 
##input layer - so we do not use backpropagation for it. 
##See Figure 3 in ex4.pdf for a diagram showing this.

# Compute deltas 

delta_1 = d2' * a_1 ; % 25 x 5000 * 5000 x 401 
delta_2 = d3' * a_2 ; % 10 x 5000 * 5000 x 26


Theta1_grad = (1/m)*delta_1; % 25 x 401
Theta2_grad = (1/m)*delta_2; % 10 x 26

% Lets Regularise the gradients 
% Setting the 1st columns of Theta1 and Theta2 to 0 since it gets multiplied with bias column of X 

Theta1(:,1) = 0; 
Theta2(:,1) = 0;
##
Theta1_grad_reg = (lambda/m) * Theta1; % 25 x 401
Theta2_grad_reg = (lambda/m) * Theta2; % 10 x 26

% add the regularised term to the unregularised theta grad 

Theta1_grad = Theta1_grad + Theta1_grad_reg;
Theta2_grad = Theta2_grad + Theta2_grad_reg;













% -------------------------------------------------------------

% =========================================================================

% Unroll gradients into vectors 
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
