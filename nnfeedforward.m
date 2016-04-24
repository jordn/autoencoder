function X = nnfeedforward(nn, x)
    X{1} = x;
    for l = 1 : numel(nn.rbm)
        X{l+1} = rbmup(nn.rbm{l}, X{l});
    end
end