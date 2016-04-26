function nn = dbnunroll(dbn)
    % Unroll a DBN ("deep belief net" - stcakde RBMs) into a NN
    nn.sizes = [dbn.sizes fliplr(dbn.sizes(1:end-1))];
    nRBMs = numel(dbn.rbm);
    for layer = 1 : numel(dbn.rbm)
        nn.rbm{layer} = dbn.rbm{layer};
        nn.rbm{layer}.deltaW = 0;
        nn.rbm{layer}.deltaB = 0;
    end
    % From code layer to 'output'
    for layer = 1 : numel(dbn.rbm)
        nn.rbm{nRBMs + layer}.W = dbn.rbm{nRBMs - layer + 1}.W';
        nn.rbm{nRBMs + layer}.a = dbn.rbm{nRBMs - layer + 1}.b;
        nn.rbm{nRBMs + layer}.b = dbn.rbm{nRBMs - layer + 1}.a;
        nn.rbm{nRBMs + layer}.deltaW = 0;
        nn.rbm{nRBMs + layer}.deltaB = 0;
        nn.rbm{nRBMs + layer}.hiddenUnits = dbn.rbm{nRBMs - layer + 1}.visibleUnits;
        nn.rbm{nRBMs + layer}.visibleUnits = dbn.rbm{nRBMs - layer + 1}.hiddenUnits;
    end
end