function [Wout,Vout, trainingError, testError ] = trainMultiLayer(Xtraining,Dtraining,Xtest,Dtest, W0, V0,numIterations, learningRate )
%TRAINMULTILAYER Trains the network (Learning)
%   Inputs:
%               X* - Trainin/test features (matrix)
%               D* - Training/test desired output of net (matrix)
%               W0 - Weights of the hidden neurons (matrix)
%               V0 - Weights of the output neurons (matrix)
%               
%               numIterations - Number of learning setps (scalar)
%               learningRate - The learningrate (scalar)
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
numTraining = size(Xtraining,2);
numTest = size(Xtest,2);
numClasses = size(Dtraining,1) - 1;
numHidden = size(W0,2);
Wout = W0;
Vout = V0;

% Calculate initial error
Ytraining = runMultiLayer(Xtraining, W0, V0);
Ytest = runMultiLayer(Xtest, W0, V0);
trainingError(1) = sum(sum((Ytraining - Dtraining).^2))/(numTraining*numClasses);
testError(1) = sum(sum((Ytest - Dtest).^2))/(numTest*numClasses);


%H(numHidden,numTraining) = 0;
%Y(numClasses,numTraining) = 0; 

% hidact = []
for n = 1:numIterations
    err = [];
    % for each obs
    for i = 1:numTraining
        % input to hidden layer
        input_bias = Xtraining(:,i);
        hidden_activation = tanh(input_bias' * Wout);
        
        
        % hidden layer to output
        hidden_bias = [1 hidden_activation];
        output_activation = hidden_bias * Vout;
        
        % back propagate 
        % error: err = T - Y
        err(:,i) = Dtraining(:,i)' - output_activation; 

        % output neurons
        delta_v = (-2) * err(:,i);
        grad_v = (-2) * err(:,i) * hidden_bias; 
        %disp(grad_v);
       
        % hidden neurons 
%        grad_w = sum(delta_v' * Vout(2:end,:)') * (1-hidden_activation.^2)' * input_bias';
        grad_w = sum((Vout(2:end,:) * delta_v)') * (input_bias * (1-hidden_activation.^2))';
        %disp(grad_w);
        
        Wout = Wout - learningRate * grad_w'; %Take the learning step.
        Vout = Vout - learningRate * grad_v'; %Take the learning step.
        
        if isnan(grad_w)
            grad_w;
        end
        
        hidden_activation = tanh(Xtraining(:,i)' * Wout);
        hidden_bias = [1 hidden_activation];
        % hidact((n - 1) * numTraining + i) = hidden_bias * hidden_bias';
        
        Ytraining = hidden_bias * Vout;
        
        hidden_activation = tanh(Xtest(:,i)' * Wout);
        hidden_bias = [1 hidden_activation];
        Ytest = hidden_bias * Vout;
        
        % sum error by all nodes
        train_obs_err(i) = sum((Ytraining - Dtraining(:,i)').^2);
        test_obs_err(i) = sum((Ytest - Dtest(:,i)').^2);
        
    end
    
    Wout;
    Vout;
    % sum error by all obs
    trainingError(1+n) = sum(train_obs_err)/(numTraining*numClasses);
    testError(1+n) = sum(test_obs_err)/(numTest*numClasses);
   
    %disp(n);
    if mod(n, 11) == 0
        subplot(2,1,1)
        plot(Wout');
        title(sprintf('Loop: %0.2f', n));
        subplot(2,1,2),
        plot(Vout);
        drawnow;
    end
end
end

