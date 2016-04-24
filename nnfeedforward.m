function y = nnfeedforward(nn, x)
    for layer = 1 : numel(nn.rbm)
        x = rbmup(nn.rbm{layer}, x);
    end
    y = x;
end