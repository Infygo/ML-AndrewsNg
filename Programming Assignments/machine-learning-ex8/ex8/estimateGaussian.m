function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% X 307 x 2 - 307 records and 2 features (latency, throughput)
% 

% Useful variables
[m, n] = size(X); % m = 307 and n = 2 

% You should return these values correctly
mu = zeros(n, 1);     % 2 x 1 
sigma2 = zeros(n, 1); % 2 x 1

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%

% idea is to compute mu. Sum of first column say latency over all the 307 records 
% Sum of 2nd column throughput over all 307 records
 
for i = 1:n
  mu(i) = sum(X(:,i));
endfor

mu = (1/m)*mu;


% Compute sigma2 i.e variance 

for i = 1:n 
  sigma2(i) = sum((X(:,i)-mu(i)).^2);
endfor

sigma2 = (1/m) * sigma2;


























% =============================================================


end
