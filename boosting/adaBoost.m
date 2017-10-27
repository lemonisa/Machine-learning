function [alphas, stumps] = adaBoost(X, y, T)
% X: data samples
% y: labels
% T: number of weak Classifiers

% initialize weights for samples
N = size(X, 2); % sample#
D = ones(1, N) / N;
aggClassification = zeros(1, N);
stumps = [];
alphas = [];
for t = 1:T
    % train single decision stump
    [epsilon, feature, polarity, threshold, classification] = decisionStump(X, y, D);
    stumps(:,t) = [feature, polarity, threshold];
    
    % calculate weights for weak classifiers
    alphas(t) = 0.5 * log((1 - epsilon) / epsilon);
    
    % update weights for samples
    D = D .* exp(-alphas(t) * (y .* classification));
    D = D./sum(D); % Normalized

end

end