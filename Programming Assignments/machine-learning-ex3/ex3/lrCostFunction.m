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
grad = zeros(size(theta)); % 400 x 1 

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


% Implement the unregularised logistic regression cost function and gradient 

% compute the z to use it for the sigmoid function 
z = X * theta ;  % 5000 x 400 * 400 x 1
predictions = sigmoid(z);  % 5000 x 1 

% compute the unregularised cost function now 
% y - 5000 x 1 
% 1 x 5000 * 5000 x 1

unreg_J = (1/m) * (-y'* log(predictions) - (1-y)' * log(1-predictions));

% compute the gradient, implement vectorisation 
% 400 x 5000 * 5000 x 1 

grad = (1/m)* X' * (predictions-y);


% regularise the cost function and gradient 
% Regularised cost function should consider theta(1) as its for the bias term X(;,1)

theta(1) = 0 ; % making it zero to eliminate its impact 
J = unreg_J + (lambda/(2*m))* sum(theta.^2);

% updating the gradient,  excluding the gradient(1) which doesnt need a regularisation
grad(2:end) = grad(2:end) + (lambda/m) * theta(2:end);









% =============================================================

grad = grad(:);

end
