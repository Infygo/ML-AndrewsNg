function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples , 12 

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta)); % 2 x 1

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Implementing the regularised linear regression - Cost function , gradient 
% m = 12, X = 12 x 2(including the bias col) , theta = 2x1 , y = 12x1

% Implementing unregularised cost function 
% Make sure to sum them over to get a scalar value as the cost function J

h_theta_x = X * theta  ; % 12 x 2 * 2 x 1

J_unreg = (1/(2*m)) * sum((h_theta_x - y).^2) ;  % sum(12 x 1 )

% Regularisation term 
% theta(1) will not be regularised , since it will be multiplied with the bias columns of 1 in X 
theta(1) = 0;
J_reg = (lambda/(2*m)) * sum((theta).^2) ;


% Final cost function J 

J = J_unreg + J_reg;


% Compute the gradient 

##thetagrad_unreg = (1/m) * sum(X(:,1)'*(h_theta_x - y)) ; 
##thetagrad_reg = (1/m) * sum(X(:,2:end)'*(h_theta_x-y)) + (lambda/m) * theta(2:end);


# Compute gradient for the entire matrix 
grad = (1/m) * X'* (h_theta_x - y) ;  % 2 x 12 * 12 x 1 

# Add regularisation terms to the columns starting from in the matrix 
grad(2:end) = grad(2:end) + (lambda/m) * theta(2:end);

% grad = [thetagrad_unreg; thetagrad_reg];














% =========================================================================

grad = grad(:);

end
