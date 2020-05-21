function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% Projecting 2D data into 1D => K =1 
% U = 2 x 2 

% You need to return the following variables correctly.
Z = zeros(size(X, 1), K); % 50 x 1 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K 
%               eigenvectors in U (first K columns). 
%               For the i-th example X(i,:), the projection on to the k-th 
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, k);
%


U_red = U(:,K); % 2 x 1 

% X - 50 x 2 
x = X'; % 2 x 50 
Z = x' * U(:,1:K);








% =============================================================

end
