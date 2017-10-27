function [ labelsOut, densityOut ] = kNN(X, k, Xt, Lt)
%KNN Your implementation of the kNN algorithm
%   Inputs:
%               X  - Features to be classified
%               k  - Number of neighbors
%               Xt - Training features
%               LT - Correct labels of each feature vector [1 2 ...]'
%
%   Output:
%               LabelsOut = Vector with the classified labels

labelsOut  = zeros(size(X,2),1);
classes = unique(Lt);
numClasses = length(classes);

train = Xt;
test = X;

% compute the distance between features

dis = [];
% 1:200 obs in train
for i = 1:size(train, 2)
    % 1:200 obs in test
    for j = 1:size(test, 2)
        % 1:2 features
        xy = 0;
        for f = 1:size(train, 1)
            ff = (train(f,i) - test(f,j))^2;
            xy = ff + xy;
            dis(i,j) = sqrt(xy);
        end
    end
end

% get index of ascending order by column
[B,I] = sort(dis);
% index for k nearest
order_k = I(1:k,:);

densityOut = cell2mat( arrayfun( @(i) mean(Lt(order_k) == classes(i))', 1:numClasses, 'UniformOutput', false))';
[m, idx] = max(densityOut, [], 1);
labelsOut = classes(idx);
return

end

