function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples #100

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));  # 3 x 1 of zeros 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
% use the advanced optimization implementation 

% compute cost function for logistic regression
% -1/m * (y*log(h(x)) + (1-y)* log(1-h(x)))

z = X*theta;

predictions = sigmoid(z);  % 100 x 1

% computing the cost function value
J = (-1/m)*((y'*log(predictions)) + ((1-y)'*log(1-predictions)));


% computing the gradient w.r.t theta 

grad(1) = (1/m)* X(:,1)' *(predictions-y);
grad(2) = (1/m)* X(:,2)' *(predictions-y);
grad(3) = (1/m)* X(:,3)' *(predictions-y);








% =============================================================

end
