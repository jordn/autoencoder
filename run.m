%% RESET
clear; clc;
cd '/Users/jordanburgess/Dropbox/MLSALT/mlsalt4 Advanced Machine Learning/autoencoder';
addpath mnist;

images = loadMNISTImages('./mnist/train-images-idx3-ubyte');
labels = loadMNISTLabels('./mnist/train-labels-idx1-ubyte');
rng(568);

%% INITIALISE
x = images(:,labels==3);

nInput = size(x, 1);
dbn.sizes = [nInput, 64, 36]; % Hidden states (square number helpful for visualisation)
opts.nEpochs = 10;
opts.nBatchSize = 128;
opts.stepSize = 0.1;
opts.momentum = 0.6;
opts.l2 = 0.00002;  % Paper subtracts 0.00002*weight from weight

for layer = 1 : numel(dbn.sizes) - 1
    dbn.rbm{layer}.W  = 0.1*randn(dbn.sizes(layer + 1), dbn.sizes(layer));
    dbn.rbm{layer}.a  = zeros(dbn.sizes(layer), 1);
    dbn.rbm{layer}.b  = zeros(dbn.sizes(layer + 1), 1);
    dbn.rbm{layer}.hiddenUnits = 'logistic';
end
dbn.rbm{numel(dbn.rbm)}.hiddenUnits = 'linear';

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
nn.sizes = [dbn.sizes fliplr(dbn.sizes(1:end-1))]
for layer = 1 : numel(dbn.rbm)
    nn.rbm{layer} = dbn.rbm{layer};
end
for layer = 1 : numel(dbn.rbm)
    nn.rbm{numel(dbn.rbm) + layer}.W = dbn.rbm{numel(dbn.rbm) - layer + 1}.W'
    nn.rbm{numel(dbn.rbm) + layer}.a = dbn.rbm{numel(dbn.rbm) - layer + 1}.b
    nn.rbm{numel(dbn.rbm) + layer}.b = dbn.rbm{numel(dbn.rbm) - layer + 1}.a
end

%% FINETUNE
x = images(:,labels==3);
nExamples = size(x,2);
kk = randperm(nExamples);
j = 1
batch = x(:, kk((j-1) * opts.nBatchSize + 1 : j * opts.nBatchSize));
x = batch;
for layer = 2 : numel(nn.rbm)
    x = rbmup(nn.rbm{layer - 1}, x);
end
y = x;
visualiselayer(batch(:,1)); title('Random reconstruction');
visualiselayer(y(:,1)); title('Random reconstruction');


