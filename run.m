clear; clc;
addpath mnist;
images = loadMNISTImages('./mnist/train-images-idx3-ubyte');
labels = loadMNISTLabels('./mnist/train-labels-idx1-ubyte');
rng(568);

X = images(:,:);

nExamples = size(X,2);
nEpochs = 100;
nBatchSize = 100;
nBatches = nExamples/nBatchSize;
nVisible = 784;
nHidden = 100; % Helpful to be a square number for visualisation for now
stepSize = 0.5;
momentum = 0.3;
deltaW = 0;

W = 0.005*randn(nHidden, nVisible); % Small random weights to break symmetry
 
for epoch = 1:nEpochs
    kk = randperm(nExamples);
    err = 0;
    for j = 1:nBatches
        batch = X(:,kk( (j-1)*nBatchSize + 1 : j*nBatchSize));
        
        % RBM Contrastive divergence (NB, not using biases yet)
        v0 = batch;           % Data (784x1)
        h0 = logistic(W*v0);  % Hidden state 0 (100x1))
        v1 = logistic(W'*h0); % Reconstruction 1 (784x1)
        h1 = logistic(W*v1);  % Hidden state 1 (100x1)
        
        deltaW = (h0*v0' - h1*v1')*stepSize/nBatchSize + momentum*deltaW;
      
        W = W + stepSize * deltaW/nBatchSize;
        err = err + sum(sum((v1 - v0).^2))/nBatchSize;
    end
    if mod(epoch,5) == 0
        fprintf('Epoch %d/%d. Reconstruction error %f (last deltaW %f)\n', epoch, nEpochs, err, sum(sum(deltaW)));
        visualiseweights(W);
        visualiselayer(v1(:,1))
        pause(1);
    end
end

visualiseweights(W);
visualiselayer(v1(:,1))