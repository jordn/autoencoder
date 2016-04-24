%% RESET
clear; clc;
cd '/Users/jordanburgess/Dropbox/MLSALT/mlsalt4 Advanced Machine Learning/autoencoder';
addpath mnist;

images = loadMNISTImages('./mnist/train-images-idx3-ubyte');
labels = loadMNISTLabels('./mnist/train-labels-idx1-ubyte');
rng(568);

%% INITIALISE
x = [images(:,labels==3) images(:,labels==7)];

nInput = size(x, 1);
dbn.sizes = [nInput, 900, 400, 225, 25]; % Hidden states (square number helpful for visualisation)
opts.nEpochs = 5;
opts.nBatchSize = 128;
opts.momentum = 0.6;
opts.l2 = 0.00002;  % Paper subtracts 0.00002*weight from weight

for layer = 1 : numel(dbn.sizes) - 1
    dbn.rbm{layer}.W  = 0.1*randn(dbn.sizes(layer + 1), dbn.sizes(layer));
    dbn.rbm{layer}.a  = zeros(dbn.sizes(layer), 1);
    dbn.rbm{layer}.b  = zeros(dbn.sizes(layer + 1), 1);
    dbn.rbm{layer}.hiddenUnits = 'logistic';
    dbn.rbm{layer}.learningRate = 0.1;
end
% Final layer is linear with Gaussian noise
dbn.rbm{numel(dbn.rbm)}.hiddenUnits = 'linear';
dbn.rbm{numel(dbn.rbm)}.learningRate = 0.001;

%% TRAIN
dbn.rbm{1} = rbmtrain(dbn.rbm{1}, x, opts);
visualiseweights(dbn.rbm{1}.W');
visualize(dbn.rbm{1}.W');

for layer = 2 : numel(dbn.rbm)
    x = rbmup(dbn.rbm{layer - 1}, x);
    dbn.rbm{layer} = rbmtrain(dbn.rbm{layer}, x, opts);
    visualiseweights(dbn.rbm{layer}.W);
end

%% UNROLL
nn.sizes = [dbn.sizes fliplr(dbn.sizes(1:end-1))];
for layer = 1 : numel(dbn.rbm)
    nn.rbm{layer} = dbn.rbm{layer};
end
for layer = 1 : numel(dbn.rbm)
    nn.rbm{numel(dbn.rbm) + layer}.W = dbn.rbm{numel(dbn.rbm) - layer + 1}.W'
    nn.rbm{numel(dbn.rbm) + layer}.a = dbn.rbm{numel(dbn.rbm) - layer + 1}.b
    nn.rbm{numel(dbn.rbm) + layer}.b = dbn.rbm{numel(dbn.rbm) - layer + 1}.a
end

%% FINETUNE
X{1} = [images(:,labels==3) images(:,labels==7) images(:,labels==5)];
for layer = 2 : numel(nn.rbm)+1
    X{layer} = rbmup(nn.rbm{layer-1}, X{layer-1});
end
kk = randperm(size(X{1},2));
i = 1;
%%
i = i +1; visualisereconstruction(X{1}(:,kk(i)), X{end}(:,kk(i)));
