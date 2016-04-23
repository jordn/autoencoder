function rbm = rbmtrain(rbm, x, opts)

nExamples = size(x,2);
nBatches = nExamples / opts.nBatchSize;

deltaW = zeros(size(rbm.W));
deltaA = zeros(size(rbm.a));
deltaB = zeros(size(rbm.b));

%%
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
            h0 = p_h0 +randn(size(rbm.b,1), opts.nBatchSize);
        else
            p_h0 = sigmoid(rbm.W * v0 + repmat(rbm.b, 1, opts.nBatchSize));
            h0 = p_h0 > rand(size(p_h0));
        end
        
        % Reconstruction 1 (784 x nBatchSize)
        v1 = sigmoid(rbm.W'*h0 + repmat(rbm.a, 1, opts.nBatchSize)); 
        
        % Hidden state 1 (100 x nBatchSize)
        p_h1 = sigmoid(rbm.W*v1 + repmat(rbm.b, 1, opts.nBatchSize, 1));
        
        deltaW = (p_h0*v0' - p_h1*v1') * opts.stepSize / opts.nBatchSize ...
            + opts.momentum * deltaW - opts.l2 * rbm.W;
        deltaA = sum(v0 - v1, 2) * opts.stepSize / opts.nBatchSize ...
            + opts.momentum * deltaA - opts.l2 * rbm.a;
        deltaB = sum(p_h0 - p_h1, 2) * opts.stepSize / opts.nBatchSize ...
            + opts.momentum * deltaB - opts.l2 * rbm.b;

        rbm.W = rbm.W + deltaW;
        rbm.a = rbm.a + deltaA;
        rbm.b = rbm.b + deltaB;
        
        err = err + sum(sum((v1 - v0).^2)) / opts.nBatchSize;
        
    end
    toc
    if mod(epoch,1) == 0
        fprintf('Epoch %d/%d. Reconstruction error %f (last deltaW %f)\n',...
            epoch, opts.nEpochs, err/nBatches, sum(sum(abs(deltaW))));
        visualiseweights(rbm.W);
        visualiselayer(v1(:,1)); title('Random reconstruction');
        pause(1);
    end
end
