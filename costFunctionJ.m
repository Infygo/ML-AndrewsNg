% X - features y - actual result theta - weights 
function J = costFunctionJ(X, y, theta)

% m - no of training records 
m = size(X, 1);

% predictions 
predictions = X * theta ;

% square errors 
square_errors = (predictions-y).^2

% cost function 
J = 1/(2*m) * sum(square_errors)