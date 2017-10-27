%% This script will help you test out your kNN code

%% Select which data to use:

% 1 = dot cloud 1
% 2 = dot cloud 2
% 3 = dot cloud 3
% 4 = OCR data

dataSetNr = 4; % Change this to load new data 

[X, D, L] = loadDataSet( dataSetNr );

% You can plot and study dataset 1 to 3 by running:
% plotCase(X,D)

%% Select a subset of the training features

numBins = 5; % Number of Bins you want to devide your data into
numSamplesPerLabelPerBin = 100; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = true; % true = select features at random, false = select the first features

[ Xt, Dt, Lt ] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom );

% Note: Xt, Dt, Lt will be cell arrays, to extract a bin from them use i.e.
% XBin1 = Xt{1};

%% Use kNN to classify data
% Note: you have to modify the kNN() function yourselfs.

% Set the number of neighbors
k = 2;

[LkNN, DkNN] = kNN(Xt{2}, k, Xt{1}, Lt{1});
plotCase(Xt{2}, DkNN)

%% Calculate The Confusion Matrix and the Accuracy
% Note: you have to modify the calcConfusionMatrix() function yourselfs.

% The confucionMatrix
cM = calcConfusionMatrix( LkNN, Lt{2})
%% 

% The accuracy
acc = calcAccuracy(cM)

%% Plot classifications
% Note: You do not need to change this code.
if dataSetNr < 4
    plotkNNResultDots(Xt{2},LkNN,k,Lt{2},Xt{1},Lt{1});
else
    plotResultsOCR( Xt{2}, Lt{2}, LkNN )
end


%%%%%%%% Cross validation for each data set
Nfolds = 5

acck = [];
for datNr = 1:4
  [X, D, L] = loadDataSet( datNr );

  numBins = Nfolds;
  numSamplesPerLabelPerBin = 100;
  selectAtRandom = true;

  [ Xt, Dt, Lt ] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom );

  for k = 2:20
    accfolds = [];
    parfor fold = 1:Nfolds
      trainX = [ Xt{setdiff(1:end,fold)} ];
      trainL = [(cellfun(@transpose, Lt, 'UniformOutput', false)){setdiff(1:end,fold)}]';
      testX = Xt{fold};
      testL = Lt{fold};

      [LkNN, DkNN] = kNN2(testX, k, trainX, trainL);
      accfolds(fold) = calcAccuracy(calcConfusionMatrix( LkNN, testL ));
      disp(sprintf('dat: %d, K: %d, fold: %d, acc: %d', datNr, k, fold, accfolds(fold)));
    end
    acck(datNr,k) = mean(accfolds);
  end
end

plot(acck(1,:),'k','linewidth',1.5)
hold on
plot(acck(2,:),'r','linewidth',1.5)
plot(acck(3,:),'g','linewidth',1.5)
plot(acck(4,:),'b','linewidth',1.5)
hold off
title('Cross validation: K versus mean accuracy')
legend('Data set 1','Data set 2','Data set 3', 'Data set 4', 'location', 'southeast')

print(sprintf('CVknn', datNr),'-dpdf');
