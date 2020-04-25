function W = randInitializeWeights(L_in, L_out)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections

%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections. 
%
%   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
%   the first column of W handles the "bias" terms
%

% You need to return the following variables correctly 
W = zeros(L_out, 1 + L_in); % weights1 25 x 401 , theta2 = 10 x 26

% ====================== YOUR CODE HERE ======================
% Instructions: Initialize W randomly so that we break the symmetry while
%               training the neural network.
%
% Note: The first column of W corresponds to the parameters for the bias unit
%
epsilon_init = 0.12;
W = rand(L_out, L_in+1) * 2 * epsilon_init - epsilon_init;



% How to choose eps_init 

##One e?ective strategy for choosing eps_init is to base it on the number of units in the network. 
##A good choice of eps_init is eps_ init = v6 vLin+Lout , 
##where Lin = sl and Lout = sl+1 are the number of units in the layers adjacent to T(l).






% =========================================================================

end
