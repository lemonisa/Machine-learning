function [best_epsilon, best_feature, best_polarity, best_threshold, best_classification] = decisionStump(X, y, D)
best_epsilon = 1;
% X: data samples
% y: labels
% D: weights for samples

F = size(X, 1); % feature#
numSteps=50.0;

% for all features
for f = 1:F
    rangeMin=min(X(f,:));
    rangeMax=max(X(f,:));
    stepSize=(rangeMax-rangeMin)/numSteps;
    
    % for all thresolds in the range of specific feature
    for i = 0:int16(numSteps)
        threshold = rangeMin + double(i)*stepSize;
        
        % for Polarity [-1,1]
        for polarity = [-1,1]
            % calculate the erro
            if(polarity == 1)                                                                         
                classification = double(X(f, :) >= threshold); % 1,0
            else
                classification = double(X(f, :) < threshold);
            end
            % convert 0 to -1
            classification(classification(:) == 0) = -1;
            
            % calculate epsilon
            epsilon = sum((classification ~= y) .* D); % D: weight
            
            % find the best weak classifier
            if(epsilon < best_epsilon)
                best_epsilon = epsilon;
                best_feature = f;
                best_polarity = polarity;
                best_threshold = threshold;
                best_classification = classification;
            end
        end
    end
end

end