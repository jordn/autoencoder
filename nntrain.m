function nn = nntrain(nn, x, targets, opts)

%% FINETUNE
% Mini-batch gradient descent with reconstruction mean squared error
nExamples = size(x,2);
nBatches = nExamples / opts.nBatchSize;

%% BACKPROP

for epoch = 1:opts.nEpochs
    kk = randperm(nExamples);
    err = 0;
    tic
    for j = 1:nBatches

        % Feed forward
        batch = x(:, kk( (j-1) * opts.nBatchSize + 1 : j * opts.nBatchSize));
        t = targets(:, kk( (j-1) * opts.nBatchSize + 1 : j * opts.nBatchSize));
        X{1} = batch;
        for layer = 2 : numel(nn.rbm)+1
            X{layer} = rbmup(nn.rbm{layer-1}, X{layer-1});
        end
        err = sum(sum(0.5*(X{end} - t).^2));

        g = X{end} - X{1}; % Gradient on the output layer.
        for layer = numel(nn.rbm):-1:1
            g = g .* X{layer+1} .* (1-X{layer+1}); % Gradient z (pre non-linearity)

            nn.rbm{layer}.deltaW = opts.learningRate * ( ...
                -g*X{layer}' / opts.nBatchSize - opts.l2 * nn.rbm{layer}.W ...
            ) + opts.momentum * nn.rbm{layer}.deltaW;

            nn.rbm{layer}.deltaB = opts.learningRate * ( ...
                -sum(g,2) / opts.nBatchSize - opts.l2 * nn.rbm{layer}.b ...
            ) + opts.momentum * nn.rbm{layer}.deltaB;

            nn.rbm{layer}.W = nn.rbm{layer}.W + nn.rbm{layer}.deltaW;
            nn.rbm{layer}.b = nn.rbm{layer}.b + nn.rbm{layer}.deltaB;
            g = nn.rbm{layer}.W'*g;
        end
        fprintf('Epoch %d/%d. Reconstruction error %f (last deltaW %f)\n',...
            epoch, opts.nEpochs, err/nBatches, sum(sum(abs(nn.rbm{layer}.deltaW))));
    end
    toc;
end

end
