function [Wout, trainingError, testError ] = trainSingleLayer(Xt,Dt,Xtest,Dtest, W0,numIterations, learningRate )
%TRAINSINGLELAYER Trains the network (Learning)
%   Inputs:
%               X* - Trainin/test features (matrix)
%               D* - Training/test desired output of net (matrix)
%               W0 - Weights of the neurons (matrix)
%
%   Output:
%               Wout - Weights after training (matrix)
%               Vout - Weights after training (matrix)
%               trainingError - The training error for each iteration
%                               (vector)
%               testError - The test error for each iteration
%                               (vector)

% Initiate variables
trainingError = nan(numIterations+1,1);
testError = nan(numIterations+1,1);
Nt = size(Xt,2); %200
Ntest = size(Xtest,2); %200
Wout = W0;

% Calculate initial error
Yt = runSingleLayer(Xt, W0);
Ytest = runSingleLayer(Xtest, W0);
trainingError(1) = sum(sum((Yt - Dt).^2))/Nt;
testError(1) = sum(sum((Ytest - Dtest).^2))/Ntest;

[numDims numTraining] = size(Xt);
numClasses = size(Dt,1);

for n = 1:numIterations
    
    err = zeros(numClasses,numTraining);
    % for each obs/instance
    for i = 1:numTraining
        % compute output: Y = W*X + b
        Y = Xt(:,i)' * Wout;
        %disp(Y)
        
        % error: err = T - Y
        err(:,i) = Dt(:,i)' - Y; 
        %disp(Dt(:,i))
        %disp(err(:,i))
        
        % update weights (delta rule): delta(W) = -(2)*(T-Y)*X
        grad_w = (-2) * err(:,i) * Xt(:,i)'; 
        %disp(grad_w)
        Wout = Wout - learningRate * grad_w';
        %disp(Wout) 
        
        % sum error by all nodes
        train_obs_err(i) = sum((Xt(:,i)' * Wout - Dt(:,i)').^2);
        test_obs_err(i) = sum((Xtest(:,i)' * Wout - Dtest(:,i)').^2);
    end
    
    % sum error by all obs
    trainingError(1+n) = sum(train_obs_err)/Nt;
    testError(1+n) = sum(test_obs_err)/Ntest;
end

end

