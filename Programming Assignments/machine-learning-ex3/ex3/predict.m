function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1); % 5000
num_labels = size(Theta2, 1); % 10 no of labels

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1); % 5000 x 1 

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


% Given theta1 and theta2 

% Layer1 - 400 units Layer2 - 25 units Layer3 - 10 units 

% using forward propogation method 
X = [ones(m, 1) X]; % 5000 x 401  , bias column 

a_1 = X;

z_2 = X * Theta1' ;  % 5000 x 401 * 401 x 25
a_2 = sigmoid(z_2); % passing this to the sigmoid function to predict i.e h(x)
% 5000 x 25 

% Now add another bias layer / column to this activation function 
a_2 = [ones(size(a_2,1),1) a_2]; % 5000 x 26 

% passing this activation function result to the output layer
z_3 = a_2 * Theta2' ; % 5000 x 26 * 26 x 10 
a_3 = sigmoid(z_3); % passing this to the sigmoid function to predict i.e h(x)
% 5000 x 10 


% Now we need to predict using the max function 

[v p] = max(a_3,[],2)  % max value in each row of the prediction 










% =========================================================================


end
