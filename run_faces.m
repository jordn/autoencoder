%% RESET
clear; clc; close all;
cd '/Users/jordanburgess/Dropbox/MLSALT/mlsalt4 Advanced Machine Learning/autoencoder';
addpath faces utils;
rng(568);
load olivettifaces_augmented

% Split data so that 25 faces for training, 5 for CV, 10 for testing.
N = size(images, 2);
trainN = 0.64 * N;
cvN = 0.12 * trainN;
testN = N - trainN - cvN;


kk = randperm(trainN);
imagesTrain = images(:, kk);
labelsTrain = labels(kk);

kk = randperm(cvN);
imagesCV = images(:, trainN + kk);
labelsCV = labels(trainN + kk);

kk = randperm(testN);
imagesTest = images(:, trainN + cvN + kk);
labelsTest = labels(trainN + + cvN + kk);

clear images labels; % Clear big data from memory;

%% INITIALISE
x = imagesTrain;
dbn.sizes = [size(x, 1), 2000, 1000, 500, 30]; % Hidden states (square number helpful for visualisation)
opts.nEpochs = 5;
opts.nBatchSize = 20;
opts.momentum = 0.6;  % Paper starts with 0.5, then switches to 0.9.
opts.l2 = 0.00002;  % Paper subtracts 0.00002*weight from weight.

for layer = 1 : numel(dbn.sizes) - 1
    dbn.rbm{layer}.W  = 0.1*randn(dbn.sizes(layer + 1), dbn.sizes(layer));
    dbn.rbm{layer}.a  = zeros(dbn.sizes(layer), 1);
    dbn.rbm{layer}.b  = zeros(dbn.sizes(layer + 1), 1);
    dbn.rbm{layer}.visibleUnits = 'logistic';
    dbn.rbm{layer}.hiddenUnits = 'logistic';
    dbn.rbm{layer}.learningRate = 0.1;
end
% Input layer is linear with Gaussian noise
dbn.rbm{1}.visibleUnits = 'linear';
dbn.rbm{1}.learningRate = 0.001;

% Code layer is linear with Gaussian noise
dbn.rbm{end}.hiddenUnits = 'linear';
dbn.rbm{end}.learningRate = 0.001;

%% TRAIN RBM
x = imagesTrain;
dbn.rbm{1} = rbmtrain(dbn.rbm{1}, x, opts);

for layer = 2 : numel(dbn.rbm)
    x = rbmup(dbn.rbm{layer - 1}, x);
    dbn.rbm{layer} = rbmtrain(dbn.rbm{layer}, x, opts);
end

%% UNROLL
nn = dbnunroll(dbn);

%% RECONSTRUCT
x = imagesCV;
X = nnfeedforward(nn, x);
kk = randperm(size(x, 2));
i = 1;
i = i +1; visualisereconstruction(X{1}(:,kk(i)), X{end}(:,kk(i))); 
mse = 255*mean( ( X{1}(:,kk(i)) - X{end}(:,kk(i)) ).^2 )

%% FINETUNE
% Mini-batch gradient descent with reconstruction mean squared error
opts.nEpochs = 30;
opts.l2 = 0.00002;
opts.nBatchSize = 1000;
opts.momentum = 0.7;  % Paper starts with 0.5, then switches to 0.9.
opts.learningRate = 0.0001; %?

x = imagesTrain;
[nn, training] = nntrain(nn, x, x, opts);

%% RECONSTRUCT (test)

nSamples = min(size(imagesTest,2), 1000);
x = imagesTest(:, 1:nSamples);

X = nnfeedforward(nn, x);
kk = randperm(size(x,2));
i = 1;
i = i +1; visualisereconstruction(X{1}(:,kk(i)), X{end}(:,kk(i)));
mse = 255*mean(mean( ( X{1} - X{end} ).^2))


%% COMPARE
nSamples = min(size(imagesTrain,2), 1000);
x = imagesTrain(:, 1:nSamples);
X = nnfeedforward(nn, x);
[mappedX, mapping] = compute_mapping(x', 'PCA', 30);
[mappedX, mapping] = compute_mapping(imagesTrain', 'PCA', 30);

for i = 1:size(x,2)
    xbar(i) = mean(x(:,i));
end
% [mappedTest, mappingTest] = compute_mapping(x', 'PCA', 30);

reconPCA = reconstruct_data(mappedX, mapping)'; % Todo. We should 'reconstruct' test data, with PCA trained on training data.
visualisecomparison(X, labelsTest, reconPCA);
% savefig('faces', gcf, 'eps');
mse = 255*mean(mean( ( X{1} - X{end} ).^2))
mse = 255*mean(mean( ( imagesTrain - reconPCA).^2))