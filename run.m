clear; clc;
% cd '/Users/jordanburgess/Dropbox/MLSALT/mlsalt4 Advanced Machine Learning/autoencoder';
addpath mnist;

images = loadMNISTImages('./mnist/train-images-idx3-ubyte');
labels = loadMNISTLabels('./mnist/train-labels-idx1-ubyte');
rng(568);

X = images(:,:);

nExamples = size(X,2);
nEpochs = 50;
nBatchSize = 100;
nBatches = nExamples/nBatchSize;
nVisible = 784;
nHidden = 100; % Helpful to be a square number for visualisation for now
stepSize = 0.1;
momentum = 0.9;
l2Penalty = 0.00002; % Paper subtracts 0.00002*weight from weight
deltaW = 0;

for epoch = 1:nEpochs
    kk = randperm(nExamples);
    err = 0;
    for j = 1:nBatches
        batch = X(:,kk( (j-1)*nBatchSize + 1 : j*nBatchSize));
        
        % RBM Contrastive divergence (NB, not using biases yet)
        v0 = batch;           % Data (784xnBatchSize)

        
        err = err + sum(sum((v1 - v0).^2))/nBatchSize;
    end
        visualiseweights(W); title('Features');
        visualiselayer(v1(:,1)); title('Random reconstruction');
        pause(1);
    end
end
visualiseweights(W);
visualiselayer(v1(:,1)); title('Random reconstruction');