function centroids = kMeansInitCentroids(X, K)
%KMEANSINITCENTROIDS This function initializes K centroids that are to be 
%used in K-Means on the dataset X
%   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
%   used with the K-Means on the dataset X
%

% You should return this values correctly
centroids = zeros(K, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should set centroids to randomly chosen examples from
%               the dataset X
%


% In practice, a good strategy for initializing the centroids is to select random examples from the training set. 

% Initialize the centroids to be random examples

% Randomly reorder the indices of examples 
randidx = randperm(size(X,1)) % Return a row vector containing a random permutation of '1:N' - 1 x 300 rowvector 

% Take the first K examples as centroids 
centroids = X(randidx(1:K),:); % returns a 3 x 2 rand intialised centroid matrix 







% =============================================================

end

