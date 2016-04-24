function nn = dbnunroll(dbn)
    % Unroll a DBN ("deep belief net" - stcakde RBMs) into a NN
    nn.sizes = [dbn.sizes fliplr(dbn.sizes(1:end-1))];
    for layer = 1 : numel(dbn.rbm)
        nn.rbm{layer} = dbn.rbm{layer};
        nn.rbm{layer}.deltaW = 0;
        nn.rbm{layer}.deltaB = 0;
    end
    for layer = 1 : numel(dbn.rbm)
        nn.rbm{numel(dbn.rbm) + layer}.W = dbn.rbm{numel(dbn.rbm) - layer + 1}.W';
        nn.rbm{numel(dbn.rbm) + layer}.a = dbn.rbm{numel(dbn.rbm) - layer + 1}.b;
        nn.rbm{numel(dbn.rbm) + layer}.b = dbn.rbm{numel(dbn.rbm) - layer + 1}.a;
        nn.rbm{numel(dbn.rbm) + layer}.deltaW = 0;
        nn.rbm{numel(dbn.rbm) + layer}.deltaB = 0;
    end
end