function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% X - 118 x 28 , y - 118 x 1 , theta - 28 x 1  

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


% Compute the cost function of Regularised logistic regression 
z = X* theta;
predictions_reg = sigmoid(z);

% Split the cost terms into 3 make sure they are summedcorrectly

J1 =  y'*log(predictions_reg);
J2 = (1-y)'*log(1-predictions_reg);

% unregularised cost fun value i.e without lambda term
unreg_J = -(1/m) * (J1+J2);

% cal of the regularisation term - make sure bias term theta0 is 
% not summed in J3 term 

theta(1) = 0 
J3 = (lambda / (2 * m))* sum(theta.^2);

% Final J term - Sum of Unregularised J and Lambda terms

J = unreg_J + J3;

grad(1) = (1/m *X(:,1)' * (predictions_reg-y));
grad(2:end) = (1/m *X(:,2:end)' * (predictions_reg-y)) + (lambda/m)*theta(2:end);




% =============================================================

end
