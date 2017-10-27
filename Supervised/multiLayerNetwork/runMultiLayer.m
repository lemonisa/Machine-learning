function [ Y, L ] = runMultiLayer( X, W, V )
%RUNMULTILAYER Calculates output and labels of the net
%   Inputs:
%               X  - Features to be classified (matrix)
%               W  - Weights of the hidden neurons (matrix)
%               V  - Weights of the output neurons (matrix)
%
%   Output:
%               Y = Output for each feature, (matrix)
%               L = The resulting label of each feature, (vector) 

S = []; %Calculate the sumation of the weights and the input signals (hidden neuron)
U = []; %Calculate the activation function as a hyperbolic tangent
Y = []; %Calculate the sumation of the output neuron

H = []; % output of hidden neuron(+ bias 1)
for i = 1:size(X,2) %1000
    S(:,i) = X(:,i)' * W;
    % disp(S(:,i))
    U(:,i) = tanh(S(:,i));
    % disp(U(:,i))
    
    %# add bias 1
    H(:,i) = [ones(1,size(U(:,i),2));U(:,i)];
    % disp(H(:,i))
    Y(:,i) = H(:,i)' * V;
    % disp(Y(:,i))
end


% Calculate classified labels
[~, L] = max(Y,[],1);
L = L(:);

end

