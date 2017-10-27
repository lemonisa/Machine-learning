%% Load face and non-face data and plot a few examples
load faces, load nonfaces
faces = double(faces); %24 24  4916 
nonfaces = double(nonfaces); %24 24  7872 

figure(1)
colormap gray
for k=1:25
subplot(5,5,k), imagesc(faces(:,:,10*k)), axis image, axis off
end

figure(2)
colormap gray
for k=1:25
subplot(5,5,k), imagesc(nonfaces(:,:,10*k)), axis image, axis off
end

%% Generate Haar feature masks
nbrHaarFeatures = 25; %% change
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);
figure(3)
colormap gray
for k = 1:25
subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2])
axis image,axis off
end

%% Create a training data set with a number of training data examples
% from each class. Non-faces = class label y=-1, faces = class label y=1
nbrTrainExamples = 2400; %% real num of observation = this value * 2
trainImages = cat(3,faces(:,:,1:nbrTrainExamples),nonfaces(:,:,1:nbrTrainExamples));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks); %25 by 4800
yTrain = [ones(1,nbrTrainExamples), -ones(1,nbrTrainExamples)]; %1 by 4800

% create test data
testImages = cat(3,faces(:,:,nbrTrainExamples+1:(nbrTrainExamples * 2)),nonfaces(:,:,nbrTrainExamples+1:(nbrTrainExamples * 2)));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks); %25 by 4800
yTest = [ones(1,nbrTrainExamples), -ones(1,nbrTrainExamples)]; %1 by 4800

%% initialize the parameters
weakClassifiers = 200;

% final classifier and check errorRate
%train_result = adaBoost(xTrain, yTrain, weakClassifiers);
[alphas, stumps] = adaBoost(xTrain, yTrain, weakClassifiers);
trainacc = runAdaBoost(alphas, stumps, xTrain, yTrain)
testacc = runAdaBoost(alphas, stumps, xTest, yTest)


%% Try out different number of weak classifiers

trainaccs = [];
testaccs = [];

for w = 2:70
    [alphas, stumps] = adaBoost(xTrain, yTrain, w);
    trainaccs(w) = runAdaBoost(alphas, stumps, xTrain, yTrain);
    testaccs(w) = runAdaBoost(alphas, stumps, xTest, yTest);
    
end

trainaccs
testaccs

plot([trainaccs; testaccs]');
title('Training (blue) and Testing (red) Accuracy vs Number of Stumps');


%% Find out a misclassified face and non-face

[alphas, stumps] = adaBoost(xTrain, yTrain, 30);
[acc2, yResults] = runAdaBoost(alphas, stumps, xTest, yTest);
misclassified = yResults ~= yTest;
tmp = testImages(:,:,misclassified);

% plot misclassified faces and non-faces
colormap gray
for k=1:25
subplot(5,5,k), imagesc(tmp(:,:,15*k)), axis image, axis off
end
