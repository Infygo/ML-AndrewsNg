function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

% Useful values
[m, n] = size(X); % 50 2 

% You need to return the following variables correctly.
U = zeros(n); % 2 x 2 
S = zeros(n); % 2 x 2 

% ====================== YOUR CODE HERE ======================
% Instructions: You should first compute the covariance matrix. Then, you
%               should use the "svd" function to compute the eigenvectors
%               and eigenvalues of the covariance matrix. 
%
% Note: When computing the covariance matrix, remember to divide by m (the
%       number of examples).
%

% Function returns parameters U and S 
% U - eignevectors , S - eigen values ,  diagonal matrix 

Sigma = (1/m) * (X'* X); % refers to the small summation symbol 

% To compute the U and S using SVD 
[U,S,V] = svd(Sigma);











% =========================================================================

end
