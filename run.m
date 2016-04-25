%% RESET
clear; clc; close all;
cd '/Users/jordanburgess/Dropbox/MLSALT/mlsalt4 Advanced Machine Learning/autoencoder';
addpath mnist utils;
rng(568);

images = loadMNISTImages('./mnist/train-images-idx3-ubyte');
labels = loadMNISTLabels('./mnist/train-labels-idx1-ubyte');
kk = randperm(size(images, 2));

N = size(images, 2);
trainN = 0.9 * N;
imagesTrain = images(:, kk(1:trainN));
labelsTrain = labels(kk(1:trainN));
imagesCV = images(:, kk(trainN+1:end));
labelsCV = labels(kk(trainN+1:end));

%% INITIALISE
x = imagesTrain;

nTrain = size(x, 1);
dbn.sizes = [nTrain, 1000, 500, 250, 2]; % Hidden states (square number helpful for visualisation)
opts.nEpochs = 10;
opts.nBatchSize = 20;
opts.momentum = 0.6;  % Paper starts with 0.5, then switches to 0.9.
opts.l2 = 0.00002;  % Paper subtracts 0.00002*weight from weight.

for layer = 1 : numel(dbn.sizes) - 1
    dbn.rbm{layer}.W  = 0.1*randn(dbn.sizes(layer + 1), dbn.sizes(layer));
    dbn.rbm{layer}.a  = zeros(dbn.sizes(layer), 1);
    dbn.rbm{layer}.b  = zeros(dbn.sizes(layer + 1), 1);
    dbn.rbm{layer}.hiddenUnits = 'logistic';
    dbn.rbm{layer}.learningRate = 0.1;
end
% Code layer is linear with Gaussian noise
dbn.rbm{end}.hiddenUnits = 'linear';
dbn.rbm{end}.learningRate = 0.001;

%% TRAIN
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

%% FINETUNE
% Mini-batch gradient descent with reconstruction mean squared error
opts.nEpochs = 3;
opts.l2 = 0.00002;
opts.learningRate = 0.01; %?

x = imagesTrain;
nn = nntrain(nn, x, x, opts);

%% RECONSTRUCT
% x = [images(:,labels==3) images(:,labels==7) images(:,labels==5)];
% labs = [labels(labels==3); labels(labels==7); labels(labels==5)];
nSamples = 1000;
x = imagesCV(:,1:nSamples);
labs = labels(1:nSamples);

X = nnfeedforward(nn, x);
kk = randperm(size(x,2));
i = 1;
i = i +1; visualisereconstruction(X{1}(:,kk(i)), X{end}(:,kk(i)));

%% VISUALISE
figure(3);
hold off;
nSamples = 400;

% Should this use RBMs with binary states etc?
for i = 0:9
    x = imagesCV(:, labelsCV==i);
    size(x)
    x = x(:, 1:nSamples);
%     X = nnfeedforward(nn, x);
    x1 = x;
    x2 = rbmupsigmoidbin(nn.rbm{1}, x1);
    x3 = rbmupsigmoidbin(nn.rbm{2}, x2);
    x4 = rbmupsigmoidbin(nn.rbm{3}, x3);
    x5 = rbmuplinear(nn.rbm{4}, x4);
    figure(3);
    s{i+1} = scatter(x5(1,:), x5(2,:));
    hold on;
    pause(2)
end
