function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
% unrolling the rolled vectors into X and theta matrices 
X = reshape(params(1:num_movies*num_features), num_movies, num_features); % 5x3
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features); % 4x3

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X)); % 5 x 3 movies x  features
Theta_grad = zeros(size(Theta)); % 4 x 3 %  users  x  features

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features 5 x 3
%        Theta - num_users  x num_features matrix of user features 4 x 3
%        Y - num_movies x num_users matrix of user ratings of movies 5 x 4
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user 5 x 4
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% grad term is the combination of X_grad and Theta_grad 

% computing the unregularised cost J 

prediction = X * Theta' ; % 5x3 * 3x4 - 5x4
prediction_rij1 = R.*prediction; % makes the elements where Rij=0 to zero in the prediction matrix 
% i.e we dont have to sum them up and making those elements to 
% zero will have the same impact 

%sum1_prediction = sum((prediction_rij1-Y).^2)

J = (1/2)* sum(sum((prediction_rij1 - Y).^2));
% first sum will give 17.99245    4.23215    0.00000    0.00000
% second sum will give sum of the row = 22.22 

% Regularisation term1 w.r.t to theta for cost function
for j=1:num_users
  reg2(j) = sum((Theta(j,:).^2),2);
endfor

reg2 = sum(reg2);
reg2 = (lambda/2)*reg2;

% Regularisation term w.r.t to x for the cost function 
for i= 1:num_movies 
  reg1(i) = sum((X(i,:).^2),2);
endfor

reg1 = sum(reg1);
reg1 = (lambda/2)*reg1;


% cost function with regularised terms 
J = J + reg1 + reg2;


% Compute the Xgradient 5x3 - movies x features
% Focus on movies 
for i = 1:num_movies
  idx = find(R(i,:)==1); % list of movies i that have been rated by user j 
  Theta_tempi = Theta(idx,:); % getting the feature values of those users whoever has rated the movie 
  Y_tempi = Y(i,idx) ; % actual ratings of movies that have been rated by user j 
  X_grad(i,:) = ((X(i,:) * Theta_tempi')-Y_tempi)*Theta_tempi;  % (5x3 * 3x4 - 5x4) * 4x3 = 5x3
endfor

% Thetagradient 4x3 - users x features 
% users 
for j = 1:num_users
  idy = find(R(:,j)==1); % list of users j that have rated movie i 5x1
  X_tempj = X(idy,:); % getting the feature values of those movies that has been rated 5x3
  Y_tempj = Y(idy,j); % actual user rating for movies that the user has rated 5x1
  Theta_grad(j,:) = ((Theta(j,:)*X_tempj')-Y_tempj')*X_tempj; % 
endfor

% adding regularisation to the gradients 
reg3 = lambda.* X;  % 5x3
reg4 = lambda.* Theta; % 4x3

X_grad = X_grad + reg3;
Theta_grad = Theta_grad + reg4;
























% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
