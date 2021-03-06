function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1); % 5000
num_labels = size(all_theta, 1); % 10

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1); % 10 x 1 

% Add ones to the X data matrix
X = [ones(m, 1) X]; % 5000 x 401;

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

% using the all_theta and X using it for the sigmoid function 
z = X * all_theta' ; % 5000 x 401 * 401 x 10 ; => 5000 x 10
prediction = sigmoid(z);

% identifying the max row value and copying into the vector p - 10x1
[v p] = max(prediction,[],2);

##The max() function returns two values.
##
##The first is the max values
##The second is the indexes of the max values.
##You want the second one as your predictions. So you can use [v p] = max(...).

##Here max(A, [], 2) means maximum of a row of the matrix A, 
##with the second dimension (2), i.e., all the columns of for a given row. 
##[] represents a matrix as opposed to a vector.








% =========================================================================


end
