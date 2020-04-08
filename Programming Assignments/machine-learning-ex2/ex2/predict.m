function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples #100

% You need to return the following variables correctly
p = zeros(m, 1); % 100 x 1 

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%

% The use of this function is to calculate the value of sigmoid function i.e
% 1./1+exp(-z) and compute the value and find if its >=0.5 or <=0.5 
% and assign 1 or 0 accordingly 

z = X*theta

##index_pos1 = find(sigmoid(z) >=0.5);  % gives the indices where the sigmoid 
##                                      % function is >=0.5 using find
##p(index_pos1) ==1 ;                   % assinging the indices in p to 1 

p(sigmoid(z)>=0.5) = 1;







% =========================================================================


end
