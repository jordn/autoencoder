function nn = nntrain(nn, x, targets, opts)

%% FINETUNE
% Mini-batch gradient descent with reconstruction mean squared error
nExamples = size(x,2);
nBatches = nExamples / opts.nBatchSize;
nLayers = numel(nn.rbm);
%% BACKPROP

for epoch = 1:opts.nEpochs
    kk = randperm(nExamples);
    tic
    for j = 1:nBatches
        % Feed forward
        batch = x(:, kk( (j-1) * opts.nBatchSize + 1 : j * opts.nBatchSize));
        t = targets(:, kk( (j-1) * opts.nBatchSize + 1 : j * opts.nBatchSize));
        X = nnfeedforward(nn, batch);
        err = sum(sum(0.5*(X{end} - t).^2));

        g = X{end} - X{1}; % Gradient on the output layer.
        for l = nLayers:-1:1
            g = g .* X{l+1} .* (1-X{l+1}); % Gradient z (pre non-linearity)

            nn.rbm{l}.deltaW = opts.learningRate * ( ...
                -g*X{l}' / opts.nBatchSize - opts.l2 * nn.rbm{l}.W ...
            ) + opts.momentum * nn.rbm{l}.deltaW;

            nn.rbm{l}.deltaB = opts.learningRate * ( ...
                -sum(g,2) / opts.nBatchSize - opts.l2 * nn.rbm{l}.b ...
            ) + opts.momentum * nn.rbm{l}.deltaB;

            nn.rbm{l}.W = nn.rbm{l}.W + nn.rbm{l}.deltaW;
            nn.rbm{l}.b = nn.rbm{l}.b + nn.rbm{l}.deltaB;
            g = nn.rbm{l}.W'*g;
        end
        fprintf('Epoch %d/%d, batch %d/%d. Reconstruction error %f (last deltaW %f)\n',...
            j, nBatches, epoch, opts.nEpochs, err/nBatches, sum(sum(abs(nn.rbm{l}.deltaW))));
    end
    toc;
end

end
