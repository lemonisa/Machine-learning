function [acc, classpred] = runAdaBoost(alpha, stumps, xTest, yTest)

    nstump = size(stumps,2);
    stumpsval = [];
    
    features = stumps(1,:);
    polarities = stumps(2,:);
    thresholds = stumps(3,:);
    for t = 1:nstump
       if(polarities(t) == 1)                                                                         
           stumpsval(t,:) = double(xTest(features(t), :) >= thresholds(t)); % 1,0
       else
           stumpsval(t,:) = double(xTest(features(t), :) < thresholds(t));
       end
    end
    % convert 0 to -1
    stumpsval(stumpsval(:) == 0) = -1; % nstump x samplesize 
    
    classpred = sign(alpha * stumpsval);
    acc = sum(classpred == yTest)/ length(yTest);
     
end