function [ Y, L ] = runSingleLayer(X, W)
%EVALUATESINGLELAYER Summary of this function goes here
%   Inputs:
%               X  - Features to be classified (matrix)
%               W  - Weights of the neurons (matrix)
%
%   Output:
%               Y = Output for each feature, (matrix)
%               L = The resulting label of each feature, (vector) 

Y = []; % 2*200
% for each obs
for i = 1:size(X,2) %200
    Y(:,i) = X(:,i)' * W;
end

% Calculate classified labels (Hint, use the max() function)   
[~, L] = max(Y,[],1);
L = L(:); %200
end

