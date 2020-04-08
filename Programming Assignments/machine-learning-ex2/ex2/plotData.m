function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
% k+ for +ve samples with 1 
% k0 for -ve samples with 0

% Plot the matrix X (exam marks) with y(0 or 1)

% find indices positions of 1 and 0's
pos = find(y==1);
neg = find(y==0); 

% plot now with those indices again X(exam1) vs X(exam2)

% plot X(exam1) X(exam2) for y=1
plot(X(pos,1),X(pos,2), "k+","markersize",7,'Linewidth',2) 

% plot X(exam1) X(exam2) for y=0
plot(X(neg,1),X(neg,2), "ko","markersize",7,'Linewidth',2,"MarkerFaceColor",'y')
 










% =========================================================================



hold off;

end
