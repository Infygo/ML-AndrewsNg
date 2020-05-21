function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1); % 3 no of centroids 


% X = X(1:3, 1:2); % 3 x 2 - check sample 

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1); % idx = 300 x 1 - 300 rows representing to which centroid each training data is nearby
% subset of {1, 2, 3}


m = size(X,1);    % 300 

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

distance_matrix = zeros(m,K); % 300 x 3  - each column holds the distance for each training data w.r.t to centroids 
% i.e column1 for distance of data points from centroid1  column2 for centroid2 distance and so on 

for i = 1:K
  Difference = bsxfun(@minus, X, centroids(i, :)); % difference between X and centroid 
  distance_matrix(:,i) = sum(Difference.^2,2);              % 
  %[x, idx] = min (distance_matrix(i, :))
endfor

for i = 1:m
  [x, idx(i)] = min (distance_matrix(i, :));
endfor



% returns idx - index of matrix for with min distance for each row i.e each training data 











% =============================================================

end

