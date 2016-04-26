function rbm = rbmtrain(rbm, x, opts)
%% Setup
nExamples = size(x,2);
nBatches = nExamples / opts.nBatchSize;

deltaW = zeros(size(rbm.W));
deltaA = zeros(size(rbm.a));
deltaB = zeros(size(rbm.b));

%% Train
for epoch = 1:opts.nEpochs
    kk = randperm(nExamples);
    err = 0;
    tic
    for j = 1:nBatches

        batch = x(:, kk( (j-1) * opts.nBatchSize + 1 : j * opts.nBatchSize));
        
        % Visible states 0, Data (784 x nBatchSize)
        v0 = batch;
        
        % Hidden states 0 (100 x nBatchSize))
        if strcmp(rbm.hiddenUnits, 'linear')
            p_h0 = rbm.W * v0 + repmat(rbm.b, 1, opts.nBatchSize);
            h0 = p_h0 + randn(size(rbm.b,1), opts.nBatchSize);
        else
            p_h0 = sigmoid(rbm.W * v0 + repmat(rbm.b, 1, opts.nBatchSize));
            h0 = p_h0 > rand(size(p_h0));
        end
        
        % Reconstruction 1 (784 x nBatchSize)
        if isfield(rbm, 'visibleUnits') && strcmp(rbm.visibleUnits, 'linear')
            p_v1 = rbm.W' * h0 + repmat(rbm.a, 1, opts.nBatchSize);
            v1 = p_v1 + randn(size(rbm.a,1), opts.nBatchSize);
        else
            v1 = sigmoid(rbm.W'*h0 + repmat(rbm.a, 1, opts.nBatchSize)); 
        end
        
        % Hidden state 1 (100 x nBatchSize)
        if strcmp(rbm.hiddenUnits, 'linear')
            p_h1 = rbm.W * v1 + repmat(rbm.b, 1, opts.nBatchSize);
        else
            p_h1 = sigmoid(rbm.W*v1 + repmat(rbm.b, 1, opts.nBatchSize, 1));
        end
        deltaW = rbm.learningRate * ( ...
                (p_h0*v0' - p_h1*v1')/ opts.nBatchSize - opts.l2 * rbm.W ...
            ) + opts.momentum * deltaW;
        deltaA = rbm.learningRate * (sum(v0 - v1, 2) / opts.nBatchSize) ...
            + opts.momentum * deltaA;
        deltaB = rbm.learningRate * (sum(p_h0 - p_h1, 2) / opts.nBatchSize) ...
            + opts.momentum * deltaB;
         
        rbm.W = rbm.W + deltaW;
        rbm.a = rbm.a + deltaA;
        rbm.b = rbm.b + deltaB;
        
        err = err + sum(sum((v1 - v0).^2)) / opts.nBatchSize;
        
    end
    toc
    fprintf('Epoch %d/%d. Reconstruction error %f (last deltaW %f)\n',...
            epoch, opts.nEpochs, err/nBatches, sum(sum(abs(deltaW))));
    if mod(epoch, 1) == 0
        visualiseweights(rbm.W); 
        visualisereconstruction(v0(:,1), v1(:,1));
        pause(0.5);
    end
end
