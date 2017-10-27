function [Wout,Vout, trainingError, testError ] = trainMultiLayer2(Xtraining,Dtraining,Xtest,Dtest, W0, V0,numIterations, learningRate )
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

passsize = 300
minerr = [testError(1)]

% hidact = []
for n = 1:numIterations
    % for each obs
    for i = 1:numTraining
        % input to hidden layer
        input_bias = Xtraining(:,i)';
        hidden_activation = tanh(input_bias * Wout);

        % hidden layer to output
        hidden_bias = [1 hidden_activation];
        output_activation = hidden_bias * Vout;

        % back propagate
        % error: err = T - Y

        % output neurons
        delta_v = (-2) * (Dtraining(:,i) - output_activation');
        grad_v = delta_v * hidden_bias;
        %disp(grad_v);

        % hidden neurons
        % grad_w = sum(delta_v' * Vout(2:end,:)') * (1-hidden_activation.^2)' * input_bias';
        grad_w = sum((Vout(2:end,:) * delta_v)') * (1-hidden_activation.^2)' * input_bias;
        %disp(grad_w);

        Wout = Wout - learningRate * grad_w'; %Take the learning step.
        Vout = Vout - learningRate * grad_v'; %Take the learning step.

        if isnan(grad_w)
          disp('NAN!')
          grad_w;
        end

        % hidden_activation = tanh(Xtraining(:,i)' * Wout);
        % hidden_bias = [1 hidden_activation];
        % hidact((n - 1) * numTraining + i) = hidden_bias * hidden_bias';
        % Ytraining = hidden_bias * Vout;
        % hidden_activation = tanh(Xtest(:,i)' * Wout);
        % hidden_bias = [1 hidden_activation];
        % Ytest = hidden_bias * Vout;
        % sum error by all nodes
        % train_obs_err(i) = sum((Ytraining - Dtraining(:,i)').^2);
        % train_obs_err(i) = sum((Ytraining - Dtraining(:,i)').^2);
        % test_obs_err(i) = sum((Ytest - Dtest(:,i)').^2);
    end

    %disp(n);
    % Dirty but faster
    trainingError(1+n) = trainingError(n);
    testError(1+n) = testError(n);

    if mod(n, 30) == 0
        Ytraining = runMultiLayer(Xtraining, Wout, Vout);
        Ytest = runMultiLayer(Xtest, Wout, Vout);
        trainingError(1+n) = sum(sum((Ytraining - Dtraining).^2))/(numTraining*numClasses);
        testError(1+n) = sum(sum((Ytest - Dtest).^2))/(numTest*numClasses);

        subplot(2,2,2);
        plot(Wout');
        title(sprintf('Hidden: %d; Loop: %0.2f; Rate: %0.10f;', numHidden, n, learningRate));
        subplot(2,2,4);
        plot(Vout);
        drawnow;
        subplot(1,2,1);
        [mErr, mErrInd] = min(testError);
        plot(trainingError,'k','linewidth',1.5)
        hold on
        plot(testError,'r','linewidth',1.5)
        plot(mErrInd,mErr,'bo','linewidth',1.5)
        hold off
        title(sprintf('Current MSE: test=%0.5f, train=%0.5f', testError(1+n), trainingError(1+n)))
        legend('Training Error','Test Error','Min Test Error')
    end

    if mod(n, passsize) == 0 && n > passsize
      passno = floor(n/passsize);
      minerr(1 + passno) = min(testError( ((passno-1) * passsize + 1): (passno * passsize) ));
      %% If error doesn't decrease more than around 0.001% in this pass,
      %% half the learning rate
      if minerr(passno) - minerr(1 + passno) <=  minerr(passno) / 8192
        learningRate = learningRate / 4;
      end
    end
end
end

